"""INDNet：光照-噪声解耦的低照度增强网络。

=============================== 版本沿革 ===============================
v1（2026-10-14）  物理反演版：enhanced = (I−N)/L + 残差
                  —— 消融单调上升，但训练集 PSNR 只有 18.7，15 张图都记不住
v2（2026-10-17）  修残差头初始化 / L 起点 / 损失权重 / 加容量
                  —— 实测 18.3765，比 v1 还低 0.06 dB，无效
v3（2026-10-17）  ★ 本版：直接输出 + 解耦辅助头
=========================================================================

为什么 v1/v2 有天花板（实测证据，勿重复踩）：
    enhanced = I / L(x) + 残差
  L 是逐像素的 sigmoid 增益图，是"最省力"的通路 —— 网络的梯度几乎全被它吸走，
  残差头贡献始终只有 1e-3 ~ 8e-3。于是模型等价于"学习到的空间变化增益"，
  只能整体提亮，**结构上无法合成细节**。
  判别实验：只用 15 张图从零训 600 步，PSNR 卡在 18.09（连记忆都做不到）。

v3 的改动：
  1. 输出改为「全局残差」：enhanced = I + delta，delta 由主干直接给。
     主干终于成为唯一的输出通路，细节合成能力不再被 L 截断。
  2. L 强制低频：在 1/4 分辨率上估计再上采样，只承担"光照"这一件事。
  3. N 与 L 保留为**辅助分支**，通过正向重构约束参与训练：
         recon = enhanced ⊙ L + N  ≈  I_low
     注意 v1 里 recon 是恒等式（R 由除法反演而来，recon ≡ I），不提供任何监督；
     v3 里 enhanced 是独立预测的，所以这一项重新变成**真实约束**，
     解耦分支也因此真正影响主干表示 —— 消融仍然有意义。

消融开关（结构不变，只关分支）：
    use_illum=F, use_noise=F : L=1, N=0 → 纯残差 U-Net（基线）
    use_illum=T, use_noise=F : 只有光照分解
    use_illum=T, use_noise=T : INDNet 完整
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-3  # 供外部复现 v1 公式时使用；v3 输出通路不再做除法


# --------------------------------------------------------------------------- #
# 基础模块
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """残差块，可指定膨胀率以扩大感受野。"""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation, bias=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class ConvBlock(nn.Module):
    """conv -> relu -> conv -> relu。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SEBlock(nn.Module):
    """通道注意力（Squeeze-and-Excitation）。"""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


def init_last_conv(module: nn.Module, std: float = 1e-2, bias: float = 0.0) -> None:
    """初始化模块里最后一个 conv。

    ⚠️ `std` 用小值而不是严格 0：权重全零时 `d loss / d input = Wᵀ·grad = 0`，
    这个头前面的层从第一步就拿不到梯度，要等末层自己先学出非零权重才"解冻"。
    """
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        return
    if std > 0:
        nn.init.normal_(last.weight, mean=0.0, std=std)
    else:
        nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.constant_(last.bias, bias)


def init_sigmoid_head(module: nn.Module, target: float = 0.9) -> None:
    """让 sigmoid 输出初始 ≈ target（默认 0.5 → L 起点偏小会整体过曝）。"""
    init_last_conv(module, std=1e-3, bias=math.log(target / (1.0 - target)))


# --------------------------------------------------------------------------- #
# 光照估计分支（低频约束）
# --------------------------------------------------------------------------- #
class IlluminationBranch(nn.Module):
    """估计光照衰减图 L ∈ (0,1)。

    ★ v3 关键：**在 1/scale 分辨率上估计，再双线性上采样回原尺寸**。
    光照本身就是低频分量，这样从结构上保证 L 不能携带细节 ——
    否则它又会变成那个"吃掉所有梯度"的捷径（v1/v2 的教训）。
    """

    def __init__(self, channels: int, num_blocks: int = 3, scale: int = 4) -> None:
        super().__init__()
        self.scale = max(1, int(scale))
        dilations = [1, 2, 4, 8][: max(1, num_blocks)]
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        ]
        layers += [ResidualBlock(channels, dilation=d) for d in dilations]
        layers += [nn.Conv2d(channels, 1, 3, 1, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        init_sigmoid_head(self.net, target=0.9)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # ★ 强制 fp32：AMP 训练时 fp16 的 sigmoid 一旦输出低于下限就变成 0，
        #   梯度随之为 0，形成"只能单向下降"的棘轮 —— 实测 L 的均值
        #   0.8553 → 0.8473 → 0.0000 单调下滑，最后永久卡死在 0。
        with torch.autocast(device_type=feat.device.type, enabled=False):
            x = F.avg_pool2d(feat.float(), self.scale) if self.scale > 1 else feat.float()
            L = self.net(x)
            # 下限用 0.02 + 0.98·σ 而不是 clamp：clamp 会把梯度直接切断，
            # 加法形式在贴到下限时仍然有梯度，不至于彻底冻死。
            L = 0.02 + 0.98 * L
        L = L.to(feat.dtype)
        return F.interpolate(L, size=feat.shape[-2:], mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
# 共享主干：4 级 U-Net
# --------------------------------------------------------------------------- #
class SharedTrunk(nn.Module):
    """编码-解码主干，输出全分辨率特征。

    4 级下采样（最大 1/8）以获得足够大的感受野；用 avg_pool + bilinear 而不是
    stride/pixel-shuffle，避免棋盘伪影（低照度增强里棋盘纹会被放大得很明显）。
    """

    def __init__(self, in_ch: int = 4, channels: int = 64, num_blocks: int = 3) -> None:
        super().__init__()
        c = channels

        self.enc1 = ConvBlock(in_ch, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 6)

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(c * 6, dilation=1 + 2 * (i % 3)) for i in range(max(1, num_blocks))],
            SEBlock(c * 6),
        )

        # 解码：上采样特征 + 同级跳跃特征，通道必须逐一对齐
        #   dec4: c*6 + c*4 = c*10 → c*4
        #   dec3: c*4 + c*2 = c*6  → c*2
        #   dec2: c*2 + c   = c*3  → c
        #   dec1: c → c（全分辨率精修，不再拼跳跃，避免重复使用同一层特征）
        self.dec4 = ConvBlock(c * 10, c * 4)
        self.dec3 = ConvBlock(c * 6, c * 2)
        self.dec2 = ConvBlock(c * 3, c)
        self.dec1 = ConvBlock(c, c)

    @staticmethod
    def _down(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor, illum: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(torch.cat([image, illum], dim=1))
        e2 = self.enc2(self._down(e1))
        e3 = self.enc3(self._down(e2))
        e4 = self.enc4(self._down(e3))
        b = self.bottleneck(e4)

        d4 = self.dec4(torch.cat([self._up(b, e3), e3], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e2), e2], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e1), e1], dim=1))
        return self.dec1(d2)


# --------------------------------------------------------------------------- #
# 主模型
# --------------------------------------------------------------------------- #
class INDNet(nn.Module):
    """光照-噪声解耦网络（v3：直接输出 + 解耦辅助头）。"""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 6,
        use_illum: bool = True,
        use_noise: bool = True,
        illum_scale: int = 4,
    ) -> None:
        super().__init__()
        self.use_illum = use_illum
        self.use_noise = use_noise
        c = base_channels

        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 消融时未启用的分支不构建，保证参数量在论文表格里真实可比
        self.illum_branch = (
            IlluminationBranch(c, num_blocks=min(4, max(1, num_blocks - 1)), scale=illum_scale)
            if use_illum
            else None
        )

        self.trunk = SharedTrunk(
            in_ch=in_channels + 1, channels=c, num_blocks=max(1, num_blocks // 2)
        )

        # 噪声头：预测加性噪声图 N（允许取负），小方差初始化
        self.head_n = (
            nn.Sequential(
                ResidualBlock(c),
                nn.Conv2d(c, in_channels, 3, 1, 1),
            )
            if use_noise
            else None
        )
        if self.head_n is not None:
            init_last_conv(self.head_n, std=1e-3)

        # 主输出头：预测对输入图的修正量 Δ（全局残差）
        self.head_out = nn.Sequential(
            ResidualBlock(c),
            SEBlock(c),
            nn.Conv2d(c, in_channels, 3, 1, 1),
        )
        init_last_conv(self.head_out, std=1e-2)

    # ------------------------------------------------------------------ #
    def forward(self, low: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.shallow(low)

        # ---- ① 光照（低频）----
        if self.use_illum:
            illum = self.illum_branch(feat)
        else:
            illum = torch.ones_like(low[:, :1])

        # ---- ② 主干 ----
        trunk_feat = self.trunk(low, illum)

        # ---- ③ 加性噪声 ----
        if self.use_noise:
            noise = self.head_n(trunk_feat)
        else:
            noise = torch.zeros_like(low)

        # ---- ④ 主输出：全局残差 I + Δ ----
        delta = self.head_out(trunk_feat)
        enhanced = low + delta

        # ---- ⑤ 正向重构约束：增强结果应当能被「光照 + 噪声」还原回输入 ----
        #      v1 里这一步是恒等式（R 由除法反演），v3 里 enhanced 是独立预测的，
        #      所以 recon ≈ low 是一条**真实约束**，把 L/N 与主输出绑在一起。
        recon = enhanced * illum + noise

        return {
            "enhanced": enhanced,
            "delta": delta,
            "reflectance": enhanced,   # 反射图即增强结果（Retinex 语义）
            "noise": noise,
            "illum": illum,
            "recon": recon,
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def enhance(self, low: torch.Tensor) -> torch.Tensor:
        """推理接口：返回裁剪到 [0,1] 的增强图像。"""
        return self.forward(low)["enhanced"].clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
def build_model(cfg: dict) -> INDNet:
    """根据配置构建模型。"""
    m = cfg.get("model", {})
    return INDNet(
        in_channels=m.get("in_channels", 3),
        base_channels=m.get("base_channels", 64),
        num_blocks=m.get("num_blocks", 6),
        use_illum=m.get("illumination_branch", True),
        use_noise=m.get("noise_branch", True),
        illum_scale=m.get("illum_scale", 4),
    )


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("configs/lol_v1.yaml", encoding="utf-8"))
    net = build_model(cfg)
    x = torch.rand(1, 3, 64, 64)
    out = net(x)
    print(f"参数量 {sum(p.numel() for p in net.parameters()) / 1e6:.3f} M")
    for k, v in out.items():
        print(f"  {k:12s} {tuple(v.shape)}  mean {v.mean():.4f}")
