import os
import sys
from pathlib import Path

import numpy as np
import torch

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working/INDNet")
ABL = Path("/kaggle/working/ablations")

# 网络结构在 2026-10-14 有重大改动，必须强制刷新代码，
# 否则工作区里的旧模型定义会与新版权重不匹配
FORCE_REFRESH = True


def find_code_root(base: Path, max_depth: int = 3):
    found = []

    def scan(folder: Path, depth: int):
        try:
            if (folder / "train.py").is_file() and (folder / "models" / "indnet.py").is_file():
                found.append(folder)
                return
        except (PermissionError, OSError):
            return
        if depth > 0:
            for c in folder.iterdir():
                if c.is_dir():
                    scan(c, depth - 1)

    scan(base, max_depth)
    return sorted(set(found), key=lambda x: len(str(x)))


def find_lol_root(base: Path, max_depth: int = 5):
    strict, loose = [], []

    def scan(folder: Path, depth: int):
        try:
            subs = [c for c in folder.iterdir() if c.is_dir()]
        except (PermissionError, OSError):
            return
        names = {c.name.lower() for c in subs}
        if {"our485", "eval15"} <= names:
            strict.append(folder)
            return
        if "low" in names and ({"high", "normal"} & names):
            loose.append(folder)
            return
        if depth > 0:
            for c in subs:
                scan(c, depth - 1)

    scan(base, max_depth)
    if strict:
        return sorted(set(strict), key=lambda x: len(str(x)))[0]
    if loose:
        return sorted({p.parent for p in loose}, key=lambda x: len(str(x)))[0]
    return None


# ---------- 准备代码 ----------
roots = find_code_root(IN)
if not roots:
    raise SystemExit("找不到代码包，请确认 Input 已挂载 indnet-code")
import shutil

if FORCE_REFRESH or not (WORK / "models" / "indnet.py").is_file():
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(roots[0], WORK)
    print("已刷新代码 ->", WORK)
else:
    print("代码已就位 ->", WORK)

os.chdir(WORK)
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

from models.indnet import INDNet  # noqa: E402

# 确认是新版模型（避免旧代码加载新权重导致形状/语义不匹配）
_src = (WORK / "models" / "indnet.py").read_text(encoding="utf-8")
if "head_n" not in _src:
    raise SystemExit("工作区代码不是物理反演版，请先更新 indnet-code 数据集（New Version）")

lol_root = find_lol_root(IN)
if lol_root is None:
    raise SystemExit("没找到 LOL 数据集")

test_low = lol_root / "eval15" / "low"
test_high = lol_root / "eval15" / "high"
if not test_low.is_dir():
    for c in lol_root.iterdir():
        if c.is_dir() and c.name.lower() == "eval15":
            test_low, test_high = c / "low", c / "high"
print("测试集:", test_low)

from PIL import Image  # noqa: E402

VARIANTS = [
    ("baseline_unet", "Baseline (U-Net)", False, False),
    ("plus_illum", "+ 光照分支", True, False),
    ("full_indnet", "INDNet (完整)", True, True),
]


def load_images(folder: Path):
    out = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            out[p.stem] = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return out


lows, highs = load_images(test_low), load_images(test_high)
common = sorted(set(lows) & set(highs))
print(f"配对 {len(common)} 张\n")


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        if mask.sum() < 16:
            return float("nan")
        d = ((a - b) ** 2)[mask]
    else:
        d = (a - b) ** 2
    mse = float(d.mean())
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def build(name: str, illum: bool, noise: bool):
    """从 checkpoint 自带的配置重建模型，避免硬编码超参对不上。"""
    ckpt = ABL / name / "best.pth"
    if not ckpt.is_file():
        return None
    rec = torch.load(ckpt, map_location="cpu")
    m = (rec.get("cfg") or {}).get("model", {})
    net = INDNet(
        in_channels=m.get("in_channels", 3),
        base_channels=m.get("base_channels", 32),
        num_blocks=m.get("num_blocks", 4),
        use_illum=m.get("illumination_branch", illum),
        use_noise=m.get("noise_branch", noise),
    )
    net.load_state_dict(rec["model"])
    net.eval()
    return net


# ============================================================
# 诊断 A：按 GT 亮度分区间的 PSNR（看分支是否专门改善暗区）
# ============================================================
print("=" * 74)
print("  诊断 A —— 按亮度分区间的 PSNR（若分支作用于暗区，这里应体现出来）")
print("=" * 74)
print(f"{'配置':<20} {'全图':>9} {'暗区(<0.3)':>12} {'中区':>9} {'亮区(>0.6)':>12}")
print("-" * 74)

for name, label, illum, nz in VARIANTS:
    net = build(name, illum, nz)
    if net is None:
        print(f"{label:<20} 权重不存在")
        continue
    acc = {"all": [], "dark": [], "mid": [], "bright": []}
    with torch.no_grad():
        for k in common:
            lo = torch.from_numpy(lows[k].transpose(2, 0, 1)).unsqueeze(0)
            pred = net.enhance(lo)[0].permute(1, 2, 0).numpy()
            gt = highs[k]
            lum = gt.mean(axis=2, keepdims=True)
            acc["all"].append(psnr(pred, gt))
            acc["dark"].append(psnr(pred, gt, np.repeat(lum < 0.30, 3, axis=2)))
            acc["mid"].append(psnr(pred, gt, np.repeat((lum >= 0.30) & (lum <= 0.60), 3, axis=2)))
            acc["bright"].append(psnr(pred, gt, np.repeat(lum > 0.60, 3, axis=2)))

    def m(x):
        v = [t for t in x if t == t and t != float("inf")]
        return float(np.mean(v)) if v else float("nan")

    print(f"{label:<20} {m(acc['all']):>9.3f} {m(acc['dark']):>12.3f} {m(acc['mid']):>9.3f} {m(acc['bright']):>12.3f}")


# ============================================================
# 诊断 B：人为加噪后重测（直接检验噪声分支的"用武之地"）
# ============================================================
print()
print("=" * 74)
print("  诊断 B —— 人为加噪后重测（噪声分支若有效，此处优势应被放大）")
print("=" * 74)
print(f"{'配置':<20} {'σ=0':>8} {'σ=0.03':>10} {'σ=0.06':>10} {'σ=0.10':>10} {'σ=0.15':>10}")
print("-" * 74)

SIGMAS = [0.0, 0.03, 0.06, 0.10, 0.15]
rng = np.random.default_rng(0)
noise_bank = {s: {k: rng.normal(0, s, lows[k].shape).astype(np.float32) for k in common} for s in SIGMAS}

for name, label, illum, nz in VARIANTS:
    net = build(name, illum, nz)
    if net is None:
        print(f"{label:<20} 权重不存在")
        continue
    row = []
    with torch.no_grad():
        for s in SIGMAS:
            vals = []
            for k in common:
                noisy = np.clip(lows[k] + noise_bank[s][k], 0, 1) if s > 0 else lows[k]
                lo = torch.from_numpy(noisy.transpose(2, 0, 1)).unsqueeze(0)
                pred = net.enhance(lo)[0].permute(1, 2, 0).numpy()
                vals.append(psnr(pred, highs[k]))
            row.append(float(np.mean(vals)))
    print(f"{label:<20} " + " ".join(f"{v:>10.3f}" for v in row))

print()
print("怎么读这两张表：")
print("  A) 若 INDNet 在『暗区』明显高于 Baseline，说明分支确实在暗光/噪声退化最重处起作用")
print("  B) 若 INDNet 在 σ 增大时优势扩大，说明噪声分支真的在做去噪 —— 这是可以直接写进论文的证据")
print("     反之，若各 σ 下三条线始终重合，说明噪声分支没有实际作用，需要改结构")
