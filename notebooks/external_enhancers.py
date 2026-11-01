"""外部（第三方）学习式增强方法 —— 与本文模型同口径对比。

为什么要接这一层
----------------
审稿人一定会问："你只和 1970 年代的 Gamma、2018 年的 CLAHE 比，
凭什么说'增强对检测无益'？" 所以必须补上**近年的学习方法**。

设计原则
--------
1. **统一接口**：每个方法都提供 `.enhance(tensor) -> tensor`，与 INDNet 对齐，
   这样检测格的 `enhance_capped()` 不用改就能直接调用。
2. **权重自动下载 + 缓存**：放到 `/kaggle/working/.ext_weights`，同一会话内只下一次。
3. **失败不致命**：任何一个方法下不下来都只打日志并跳过，不影响其他方法评测。

当前支持
--------
- `zero_dce`      CVPR 2020，权重在官方 GitHub，最稳
- `retinexformer` ICCV 2023，需要克隆仓库 + Google Drive 权重（第二阶段接入）

用法（在检测格的环境变量里）
---------------------------
    os.environ["EXDARK_EXTRA"] = "zero_dce"
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WD = Path("/kaggle/working/.ext_weights")

ZERO_DCE_URLS = [
    "https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth",
    "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth",
    "https://ghproxy.net/https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth",
]

# Retinexformer（ICCV 2023）：官方权重只在百度网盘/Google Drive 文件夹里，
# Kaggle 上很难拿。这里改用 jingxiLiao/GT-mean-loss 这个 fork ——
# 它把 **权重和架构文件都提交进了 GitHub 仓库**，可以直接 raw 下载。
# 已本地验证：n_feat=40 / stage=1 / num_blocks=[1,2,2] ⇒ 1.61 M 参数，strict 加载 0 缺 0 多。
_RF_REPO = "https://raw.githubusercontent.com/jingxiLiao/GT-mean-loss/main/"
RF_ARCH_URLS = [
    "https://ghproxy.net/" + _RF_REPO + "basicsr/models/archs/RetinexFormer_arch.py",
    _RF_REPO + "basicsr/models/archs/RetinexFormer_arch.py",
]
RF_WEIGHT_URLS = [
    "https://ghproxy.net/" + _RF_REPO + "pretrained_weights/LOL_v1.pth",
    _RF_REPO + "pretrained_weights/LOL_v1.pth",
]


# --------------------------------------------------------------------------- #
def _download(urls: list[str], dst: Path, name: str) -> bool:
    """按顺序尝试下载，成功返回 True。已有且非空则直接复用。"""
    if dst.is_file() and dst.stat().st_size > 1024:
        print(f"  [缓存] {name}: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    for u in urls:
        try:
            print(f"  [下载] {name} <- {u}")
            urllib.request.urlretrieve(u, dst)
            if dst.is_file() and dst.stat().st_size > 1024:
                print(f"  [完成] {name}: {dst.stat().st_size / 1e6:.1f} MB")
                return True
            print("  [失败] 文件过小，换下一个源")
        except Exception as exc:                      # noqa: BLE001
            print(f"  [失败] {type(exc).__name__}: {exc}")
    return False


# --------------------------------------------------------------------------- #
class ZeroDCE(nn.Module):
    """Zero-DCE（CVPR 2020）的 LE-Net。

    结构完全按官方实现（7 层全卷积 + 跳跃拼接），输出 8 组曲线的参数图，
    迭代 8 次 LE 曲线： x_{n+1} = x_n + r_n ⊙ (x_n² − x_n)

    参数名与官方 checkpoint 保持一致（e_conv1 … e_conv7），可直接 load_state_dict。
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        w = width
        self.e_conv1 = nn.Conv2d(3, w, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(w, w, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(w, w, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(w, w, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(w * 2, w, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(w * 2, w, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(w * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))
        maps = torch.split(r, 3, dim=1)

        out = x
        for i in range(8):
            out = out + maps[i] * (out * out - out)
        return out

    @torch.no_grad()
    def enhance(self, t: torch.Tensor) -> torch.Tensor:
        return self.forward(t).clamp(0.0, 1.0)


def _load_state_dict(path: Path) -> dict:
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location="cpu")
    # 兼容两种常见的包裹：{"state_dict": ...}（basicSR）与 {"params": ...}（Retinexformer）
    if isinstance(sd, dict):
        for key in ("state_dict", "params"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    # 兼容 DataParallel 保存的 "module." 前缀
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _ensure_einops() -> bool:
    """Retinexformer 的架构文件依赖 einops（纯 Python，装了没副作用）。"""
    try:
        import einops  # noqa: F401
        return True
    except ImportError:
        pass
    import subprocess
    import sys

    print("  [安装] einops ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops"], check=False)
    try:
        import einops  # noqa: F401
        return True
    except ImportError:
        return False


class _EnhWrapper(nn.Module):
    """把只提供 forward 的第三方模型包成统一的 .enhance() 接口。

    ⚠️ 必须 pad 到 8 的倍数：Retinexformer 的 UNet 解码器用 stride-2 池化 +
    双倍上采样，**奇数边长会错一格**（实测 427 → 池化得 213 → 上采样回 426，
    与跳跃连接的 427 对不上，报 "Expected size 186 but got size 187"）。
    ExDark 里 640×427 / 507×338 这类尺寸很多，不 pad 会直接崩。
    用 replicate 而不是 reflect —— 小图上 reflect 要求 pad < dim，容易越界。
    """

    MULTIPLE = 8

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def enhance(self, t: torch.Tensor) -> torch.Tensor:
        h, w = t.shape[-2:]
        m = self.MULTIPLE
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        y = self.net(t)
        return y[..., :h, :w].clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
def build(name: str, device: str = "cuda"):
    """按名字构建增强器；返回 None 表示这个会话里拿不到（不影响其他方法）。"""
    key = name.strip().lower()

    if key == "zero_dce":
        p = WD / "Epoch99.pth"
        if not _download(ZERO_DCE_URLS, p, "zero_dce"):
            print("  [!] zero_dce 权重下载失败，跳过该方法")
            return None
        net = ZeroDCE()
        try:
            net.load_state_dict(_load_state_dict(p), strict=True)
        except Exception as exc:                      # noqa: BLE001
            print(f"  [!] zero_dce 权重与结构不匹配：{type(exc).__name__}: {exc}")
            print("      可能是官方仓库改了文件名，检查 external_enhancers.py 里的 URL")
            return None
        return net.to(device).eval()

    if key == "retinexformer":
        if not _ensure_einops():
            print("  [!] einops 装不上，跳过 retinexformer")
            return None

        arch_path = WD / "RetinexFormer_arch.py"
        w_path = WD / "LOL_v1_rf.pth"
        if not _download(RF_ARCH_URLS, arch_path, "rf_arch"):
            print("  [!] retinexformer 架构文件下载失败，跳过")
            return None
        if not _download(RF_WEIGHT_URLS, w_path, "rf_lol_v1"):
            print("  [!] retinexformer 权重下载失败，跳过")
            return None

        import importlib.util

        try:
            spec = importlib.util.spec_from_file_location("retinexformer_arch", arch_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # 超参由权重键的形状反推：conv1 = [40, 4, 1, 1] ⇒ n_feat=40；
            # body 只有 index 0 ⇒ stage=1；每层块数 1 / 2 / 2。
            net = mod.RetinexFormer(in_channels=3, out_channels=3,
                                    n_feat=40, stage=1, num_blocks=[1, 2, 2])
            net.load_state_dict(_load_state_dict(w_path), strict=True)
        except Exception as exc:                      # noqa: BLE001
            print(f"  [!] retinexformer 构建失败：{type(exc).__name__}: {exc}")
            return None

        print(f"  [OK] retinexformer 已加载（{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M）")
        return _EnhWrapper(net).to(device).eval()

    print(f"  [!] 未知的外部增强方法：{name}")
    return None
