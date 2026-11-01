"""损失函数。

总损失由四项构成：

    L_total = w_rec      · ( L_recon_low  + L_recon_high )
            + w_smooth   · L_illum_smooth
            + w_noise    · L_noise_consistency
            + w_perceptual · L_perceptual

其中：
    L_recon_low    : || I_low - (R + N) ⊙ L ||   —— Retinex 分解自洽性
    L_recon_high   : || R - I_high ||            —— 主监督信号（增强结果要对得上参考图）
    L_illum_smooth : 光照图全变分平滑（用反射图梯度做引导）—— 抑制光照伪影
    L_noise_consistency : 约束噪声图落在正确的区域 —— 本文的"解耦"证据
    L_perceptual   : VGG 感知损失（可选）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 基础损失
# --------------------------------------------------------------------------- #
class CharbonnierLoss(nn.Module):
    """Charbonnier (smooth L1 的平滑版)，对异常值更稳健，复原任务常用。"""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class IlluminationSmoothnessLoss(nn.Module):
    """光照图平滑损失（Retinex-Net 风格）。

    直接在 L 上做全变分会连同真实的光照边界一起抹掉，所以用反射图 R 的梯度做门控：
        L = mean( |∇L| · exp(-λ|∇R|) )
    在反射图纹理丰富的地方（真实边缘）放松平滑约束，在平坦区加强平滑。
    """

    def __init__(self, lambda_grad: float = 10.0) -> None:
        super().__init__()
        self.lam = lambda_grad

    def forward(self, illum: torch.Tensor, reflectance: torch.Tensor) -> torch.Tensor:
        dx_l = torch.abs(illum[:, :, :, :-1] - illum[:, :, :, 1:])
        dy_l = torch.abs(illum[:, :, :-1, :] - illum[:, :, 1:, :])

        # 反射图梯度取通道平均，作为边缘指示
        dx_r = torch.mean(torch.abs(reflectance[:, :, :, :-1] - reflectance[:, :, :, 1:]), dim=1, keepdim=True)
        dy_r = torch.mean(torch.abs(reflectance[:, :, :-1, :] - reflectance[:, :, 1:, :]), dim=1, keepdim=True)

        wx = torch.exp(-self.lam * dx_r)
        wy = torch.exp(-self.lam * dy_r)

        loss = torch.mean(wx * dx_l) + torch.mean(wy * dy_l)
        return loss


class NoiseConsistencyLoss(nn.Module):
    """噪声一致性损失 —— 本文解耦设计的关键监督。

    在正确的退化模型 I = R⊙L + N 下，若反射图 R 已接近参考图 I_high，
    则有 N = I_low − I_high⊙L。本损失就是把噪声图拉向这个目标。

    支持两种模式：

    mode="residual"（默认，推荐）
        监督噪声图逼近"去掉光照并扣掉正常光内容后的残差"：
            target = I_low − I_high ⊙ L
        它教会网络"哪些像素该被归到噪声图里"。

    mode="dark_focus"
        只做区域约束：噪声应集中在光照弱的区域。
            L = mean( |N| · L )
        不含监督信息，纯正则，适合作为附加约束。

    实现备注：target 必须 detach，否则损失会通过 L 反向传播，
    退化成"让 L 去迎合 N"的平凡解。
    """

    def __init__(self, mode: str = "residual", eps: float = 1e-3) -> None:
        super().__init__()
        if mode not in ("residual", "dark_focus"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.eps = eps
        self.charb = CharbonnierLoss()

    def forward(
        self,
        noise: torch.Tensor,
        illum: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "dark_focus":
            return torch.mean(torch.abs(noise) * illum)

        target = (low - high * illum).detach()
        # 数值兜底：光照图接近 0 时目标值本身是合理的，保守裁剪防异常
        target = torch.clamp(target, -1.0, 1.0)
        return self.charb(noise, target)


class PerceptualLoss(nn.Module):
    """VGG16 感知损失（可选）。

    torchvision 的预训练权重需要联网下载；下载失败时自动静默跳过，
    不会让训练中断。
    """

    def __init__(
        self,
        layers: tuple[int, ...] = (3, 8, 15),
        weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.enabled = True
        self.layer_ids = layers
        self.weights = weights
        self.loss_fn = nn.L1Loss()
        self.slices: nn.ModuleList | None = None
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _build(self, device: torch.device) -> None:
        """把 VGG 切成若干段并冻结，避免重复计算中间特征。"""
        try:
            from torchvision.models import VGG16_Weights, vgg16

            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
            for p in vgg.parameters():
                p.requires_grad = False

            slices = nn.ModuleList()
            start = 0
            for end in self.layer_ids:
                slices.append(nn.Sequential(*[vgg[i] for i in range(start, end + 1)]))
                start = end + 1

            self.slices = slices
            self._mean = self._mean.to(device)
            self._std = self._std.to(device)
        except Exception as exc:  # pragma: no cover - 依赖网络下载
            print(f"[warn] 感知损失不可用（{exc}），已自动跳过该项。")
            self.enabled = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return pred.new_zeros(())
        if self.slices is None:
            self._build(pred.device)
        if not self.enabled:
            return pred.new_zeros(())

        x = (pred - self._mean) / self._std
        y = (target - self._mean) / self._std

        loss = pred.new_zeros(())
        for seg, w in zip(self.slices, self.weights):
            x, y = seg(x), seg(y)
            loss = loss + w * self.loss_fn(x, y)
        return loss


# --------------------------------------------------------------------------- #
# 组合损失
# --------------------------------------------------------------------------- #
class INDNetLoss(nn.Module):
    """把各分项按权重加起来，并返回明细方便观察训练动态。"""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        lc = cfg.get("loss", {})
        mc = cfg.get("model", {})

        # 消融开关：未启用的分支不能施加对应损失，否则会产生互相矛盾的目标
        self.use_illum = bool(mc.get("illumination_branch", True))
        self.use_noise = bool(mc.get("noise_branch", True))

        self.w_rec = float(lc.get("w_rec", 1.0))
        # 正向重构约束 enhanced⊙L + N ≈ I_low。
        # ★ v3 起这一项是**真实约束**：enhanced 由主干独立预测，recon 不再恒等于 I
        #   （v1/v2 里 R 由 (I−N)/L 反演而来，recon ≡ I，该项提供不了任何信号）。
        #   它把光照/噪声两个辅助分支与主输出绑在一起，消融因此仍然有意义。
        self.w_recon_low = float(lc.get("w_recon_low", 1.0))
        # v1/v2 的「物理反演结果 R 也要对齐参考图」。v3 里 reflectance 就是 enhanced，
        # 再加一项等于把主监督权重翻倍，故保留键位做兼容但不再使用。
        self.w_refl = float(lc.get("w_refl", 0.0))
        self.w_smooth = float(lc.get("w_illum_smooth", 0.005))
        self.w_noise = float(lc.get("w_noise_consistency", 0.05))
        # ★ v3.1：噪声图幅度正则。
        # 没有它，重构约束 `enhanced⊙L + N ≈ I_low` 里的 N（3 通道自由图）会把
        # **所有空间变化都吸收掉**，最优解退化成"L 取常数、N 背锅" ——
        # 实测 2026-10-19：L 的空间 std = 0.0000，光照分支彻底失效，消融三行会完全重合。
        # 给 N 加 L1 惩罚后，空间变化只能由 L 的乘性项解释，L 才必须真的干活。
        self.w_noise_l1 = float(lc.get("w_noise_l1", 0.02))
        # ★ v3.2：光照图的直接监督（配对数据的亮度比，且只在低频上比）。
        # 为什么必须加：只靠重构约束 `enhanced⊙L + N ≈ I_low` **约束不住 L** ——
        # N 是 3 通道自由图，实测 |N| 会长到 0.35（比输入信号还大）把重构误差全吃掉，
        # L 于是保持成常数图（空间 std = 0.0000），光照分支实际失效，消融三行会重合。
        # 目标取「低照度 / 正常光」的比值再强模糊到 1/8 分辨率：
        #   · 低频 = 光照衰减 → 监督 L
        #   · 高频（含噪声）= low − enhanced⊙L → 由噪声分支通过重构项解释
        # 这样两个分支才有各自的分工，消融才有意义。
        self.w_illum_sup = float(lc.get("w_illum_sup", 1.0))
        self.illum_sup_blur = int(lc.get("illum_sup_blur", 8))
        self.w_perceptual = float(lc.get("w_perceptual", 0.0))

        self.charb = CharbonnierLoss()
        self.smooth = IlluminationSmoothnessLoss()
        self.noise_cons = NoiseConsistencyLoss(mode=lc.get("noise_mode", "residual"))
        self.perceptual = PerceptualLoss() if self.w_perceptual > 0 else None

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        enhanced = outputs["enhanced"]
        reflectance = outputs["reflectance"]
        noise = outputs["noise"]
        illum = outputs["illum"]
        recon = outputs["recon"]
        zero = reflectance.new_zeros(())

        # ① 主监督：最终输出对齐参考图
        l_out = self.charb(enhanced, high)
        # ② v1/v2 遗留项；v3 里 reflectance ≡ enhanced，w_refl 默认 0（加了等于翻倍主监督）
        l_refl = self.charb(reflectance, high) if self.w_refl > 0 else zero
        # ③ 正向重构 —— 把光照/噪声分支与主输出绑在一起。
        #    两个分支全关时 recon ≡ enhanced，约束会与主监督直接冲突，故跳过。
        l_recon_low = (
            self.charb(recon, low) if (self.use_illum or self.use_noise) else zero
        )
        # ④ 光照平滑 —— 光照分支关闭时 illum 恒为 1，该项无意义
        l_smooth = self.smooth(illum, reflectance) if self.use_illum else zero
        # ⑤ 噪声一致性 —— 需要光照图给出目标、噪声图给出被约束对象，缺一不可
        l_noise = (
            self.noise_cons(noise, illum, low, high)
            if (self.use_illum and self.use_noise)
            else zero
        )
        # ⑥ 噪声幅度正则 —— 防止 N 变成"第二张残差图"把 L 要干的活全抢走
        l_noise_l1 = noise.abs().mean() if self.use_noise else zero
        # ⑦ 光照监督 —— 低频亮度比。没有它 L 会塌缩成常数（见 __init__ 的说明）
        if self.use_illum and self.w_illum_sup > 0:
            tgt = (low / (high + 0.05)).clamp(0.02, 1.0).detach()
            b = max(1, self.illum_sup_blur)
            if min(tgt.shape[-2:]) >= b:
                tgt = F.avg_pool2d(tgt, b)
            tgt = F.interpolate(tgt, size=illum.shape[-2:], mode="bilinear", align_corners=False)
            l_illum_sup = self.charb(illum, tgt)
        else:
            l_illum_sup = zero

        total = (
            self.w_rec * l_out
            + self.w_refl * l_refl
            + self.w_recon_low * l_recon_low
            + self.w_smooth * l_smooth
            + self.w_noise * l_noise
            + self.w_noise_l1 * l_noise_l1
            + self.w_illum_sup * l_illum_sup
        )

        detail = {
            "out": float(l_out.detach()),
            "refl": float(l_refl.detach()),
            "recon_low": float(l_recon_low.detach()),
            "smooth": float(l_smooth.detach()),
            "noise": float(l_noise.detach()),
            "noise_l1": float(l_noise_l1.detach()),
            "illum_sup": float(l_illum_sup.detach()),
        }

        if self.perceptual is not None and self.w_perceptual > 0:
            l_perc = self.perceptual(enhanced, high)
            total = total + self.w_perceptual * l_perc
            detail["perceptual"] = float(l_perc.detach())

        detail["total"] = float(total.detach())
        return total, detail
