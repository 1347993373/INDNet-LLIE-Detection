"""拟合体检：训练集 vs 测试集 PSNR —— 判断是欠拟合、过拟合，还是模型根本没学到东西。

粘进 Kaggle 一格直接跑（只推理，1–2 分钟）。也可本地跑（在 INDNet/ 根目录下）。

输出四组数字：
  ① 恒等基线    原图直接当输出能得多少 dB（LOL-v1 约 13–14）。模型减它才是"真学到多少"
  ② 训练集 PSNR 完整图、不做增强、取前 N 张（默认 100）
  ③ 测试集 PSNR 全部 15 张
  ④ 光照图统计  L 的均值 / 空间标准差。标准差接近 0 = 光照分支塌缩成常数，等于没在估光照

判读：
  · 训练 25+、测试 18 → 过拟合（数据量/正则问题）
  · 训练也 18 左右    → 欠拟合（容量 / 优化 / 损失设计问题）★ 本次最可能
  · 只有比恒等基线高 2 dB 以内 → 网络基本没学到东西

可用环境变量覆盖（默认值适配 Kaggle）：
  FIT_CKPT / FIT_CODE_ROOT / FIT_DATA / FIT_N
"""

import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

CKPT = Path(os.environ.get("FIT_CKPT", "/kaggle/working/ablations/full_indnet/best.pth"))
CODE_ROOT_HINT = Path(os.environ.get("FIT_CODE_ROOT", "/kaggle/input"))
DATA_OVERRIDE = os.environ.get("FIT_DATA", "")
N_TRAIN = int(os.environ.get("FIT_N", "100"))
WORK = Path("/kaggle/working/INDNet")


# --------------------------------------------------------------------------- #
# 1. 定位代码
# --------------------------------------------------------------------------- #
def _looks_like_code_root(p: Path) -> bool:
    try:
        return (p / "train.py").is_file() and (p / "models" / "indnet.py").is_file()
    except (PermissionError, OSError):
        return False


def _find_code_root(base: Path, max_depth: int = 3) -> Path | None:
    found: list[Path] = []

    def scan(folder: Path, depth: int) -> None:
        try:
            if _looks_like_code_root(folder):
                found.append(folder)
                return
        except (PermissionError, OSError):
            return
        if depth > 0:
            try:
                for c in folder.iterdir():
                    if c.is_dir():
                        scan(c, depth - 1)
            except (PermissionError, OSError):
                pass

    scan(base, max_depth)
    if not found:
        return None
    # 路径短的优先；排除工作区里那份大写 INDNet 的旧副本
    found.sort(key=lambda x: ("INDNet" in str(x), len(str(x))))
    return found[0]


root = None
if _looks_like_code_root(Path.cwd()):
    root = Path.cwd()
else:
    src = _find_code_root(CODE_ROOT_HINT)
    if src is None:
        raise SystemExit(
            f"找不到代码根（在 {CODE_ROOT_HINT} 下找 train.py + models/indnet.py）。"
            "请确认 indnet-code 数据集已挂载。"
        )
    shutil.copytree(src, WORK, dirs_exist_ok=True)
    root = WORK

os.chdir(root)
sys.path.insert(0, str(root))
print(f"代码根 -> {root}")

from datasets.lol_dataset import PairedLowLightDataset, resolve_pair_dirs  # noqa: E402
from models.indnet import build_model  # noqa: E402
from utils.metrics import calculate_psnr, calculate_ssim  # noqa: E402

if not CKPT.is_file():
    raise SystemExit(f"权重不存在：{CKPT}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# 2. 载入权重（用 checkpoint 里存的 cfg，保证结构完全一致）
# --------------------------------------------------------------------------- #
try:
    rec = torch.load(CKPT, map_location="cpu", weights_only=False)
except TypeError:
    rec = torch.load(CKPT, map_location="cpu")

cfg = rec.get("cfg") or {}
if not cfg:
    raise SystemExit("checkpoint 里没有 cfg，无法还原模型结构")

if DATA_OVERRIDE:
    cfg.setdefault("data", {})["root"] = DATA_OVERRIDE

tr_cfg, da_cfg = cfg.get("train") or {}, cfg.get("data") or {}
print(
    "权重配方 total_iter / epochs / patch / batch = "
    f"({tr_cfg.get('total_iter', 0)}, {tr_cfg.get('epochs', 0)}, "
    f"{da_cfg.get('patch_size', 0)}, {da_cfg.get('batch_size', 0)})"
)
print(f"best_psnr = {rec.get('best_psnr')}   epoch = {rec.get('epoch')}")

model = build_model(cfg).to(device)
model.load_state_dict(rec["model"])
model.eval()
n_par = sum(p.numel() for p in model.parameters()) / 1e6
print(f"参数量 {n_par:.3f} M | 设备 {device}")


# --------------------------------------------------------------------------- #
# 3. 数据：训练集用完整图、不做增强（与测试集同口径才可比）
# --------------------------------------------------------------------------- #
def find_lol_root() -> Path:
    """定位 LOL 数据根。

    Kaggle 的数据集挂载前缀会变（`/kaggle/input/<slug>`、
    `/kaggle/input/datasets/<owner>/<slug>` …），而 checkpoint 里存的是**训练当时**的
    绝对路径，换个会话就很可能失效。所以先试配置里的，取不到就在 /kaggle/input 下搜。
    """
    cand = Path(cfg["data"]["root"])
    if (cand / "our485").is_dir() or (cand / "eval15").is_dir():
        return cand

    hits: list[Path] = []
    for low in glob.glob("/kaggle/input/**/our485/low", recursive=True):
        r = Path(low).parent.parent
        if (r / "eval15").is_dir():
            hits.append(r)
    hits = sorted(set(hits), key=lambda p: len(str(p)))
    if hits:
        print(f"[!] 配置里的数据根不存在：{cand}\n    自动改用：{hits[0]}")
        return hits[0]

    raise SystemExit(
        f"找不到 LOL 数据根：{cand} 不存在，/kaggle/input 下也没有含 our485+eval15 的目录。\n"
        "请确认 lol-dataset 已挂载到 Notebook 的 Input。"
    )


root_dir = find_lol_root()
print(f"数据根 -> {root_dir}")
train_low, train_high = resolve_pair_dirs(root_dir, "train")
test_low, test_high = resolve_pair_dirs(root_dir, "test")
train_set = PairedLowLightDataset(train_low, train_high, patch_size=None, augment=False)
test_set = PairedLowLightDataset(test_low, test_high, patch_size=None, augment=False)
print(f"训练集 {len(train_set)} 张（取前 {min(N_TRAIN, len(train_set))}）| 测试集 {len(test_set)} 张")


# --------------------------------------------------------------------------- #
# 4. 评测
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(dataset, limit: int | None = None) -> dict:
    n = len(dataset) if limit is None else min(limit, len(dataset))
    ps, ss, ident, illum_mean, illum_std, resid = [], [], [], [], [], []

    for i in range(n):
        item = dataset[i]
        low = item["low"].unsqueeze(0).to(device)
        high = item["high"].unsqueeze(0).to(device)

        out = model(low)
        enh = out["enhanced"].clamp(0.0, 1.0)

        pred = enh[0].permute(1, 2, 0).float().cpu().numpy()
        gt = high[0].permute(1, 2, 0).float().cpu().numpy()
        idt = low[0].permute(1, 2, 0).float().cpu().numpy()

        ps.append(calculate_psnr(pred, gt))
        ss.append(calculate_ssim(pred, gt))
        ident.append(calculate_psnr(idt, gt))

        L = out["illum"]
        illum_mean.append(float(L.mean()))
        illum_std.append(float(L.std(dim=(2, 3)).mean()))  # 每张图的空间标准差
        resid.append(float(out["delta"].abs().mean()) if "delta" in out
                     else float((enh - out["reflectance"]).abs().mean()))

        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1}/{n}")

    return {
        "n": n,
        "psnr": float(np.mean(ps)),
        "ssim": float(np.mean(ss)),
        "identity": float(np.mean(ident)),
        "illum_mean": float(np.mean(illum_mean)),
        "illum_std": float(np.mean(illum_std)),
        "residual": float(np.mean(resid)),
    }


print("\n" + "=" * 72)
print(f"  测试集（{len(test_set)} 张，完整图）")
print("=" * 72)
te = evaluate(test_set)
print(f"  PSNR {te['psnr']:.4f} | SSIM {te['ssim']:.4f} | 恒等基线 {te['identity']:.4f}")

print("\n" + "=" * 72)
print(f"  训练集（前 {min(N_TRAIN, len(train_set))} 张，完整图，无增强）")
print("=" * 72)
tr = evaluate(train_set, limit=N_TRAIN)
print(f"  PSNR {tr['psnr']:.4f} | SSIM {tr['ssim']:.4f} | 恒等基线 {tr['identity']:.4f}")


# --------------------------------------------------------------------------- #
# 5. 参照基线：一个常数增益 a·I 能拿到多少 dB
# --------------------------------------------------------------------------- #
print("\n" + "=" * 72)
print("  参照：单一全局亮度系数 a·I 的最优 PSNR")
print("=" * 72)
best_a, best_p = 1.0, -1e9
for a in np.arange(1.0, 10.01, 0.5):
    ps = []
    for i in range(len(test_set)):
        item = test_set[i]
        low = item["low"].permute(1, 2, 0).numpy()
        gt = item["high"].permute(1, 2, 0).numpy()
        ps.append(calculate_psnr(np.clip(low * a, 0, 1), gt))
    p = float(np.mean(ps))
    if p > best_p:
        best_a, best_p = float(a), p
print(f"  最优 a = {best_a:.1f}  →  PSNR {best_p:.4f}")
print(f"  模型比全局增益高 {te['psnr'] - best_p:+.4f} dB")


# --------------------------------------------------------------------------- #
# 6. 判读
# --------------------------------------------------------------------------- #
gap = tr["psnr"] - te["psnr"]
gain = te["psnr"] - te["identity"]

print("\n" + "=" * 72)
print("  体检结论")
print("=" * 72)
print(f"  训练集 - 测试集 = {gap:+.4f} dB")
print(f"  测试集 - 恒等基线 = {gain:+.4f} dB")
print(f"  测试集 - 最优全局增益 = {te['psnr'] - best_p:+.4f} dB")
print(f"  光照图 L: 均值 {te['illum_mean']:.4f} | 空间标准差 {te['illum_std']:.4f}")
print(f"  残差头贡献 |delta| = {te['residual']:.4f}")
print()

if gap > 2.0:
    print("→ 过拟合：训练集明显高于测试集。15 张测试图方差大，也可能有一部分是划分运气。")
    print("   方向：加强正则 / 更多数据增广，而不是继续加大模型。")
elif tr["psnr"] < 20.0:
    print("→ 欠拟合：训练集本身就很低，说明模型连训练数据都没拟合好。")
    print("   这不是训练量问题（6000 → 60000 iter 只涨了 0.09 dB），优先级：")
    print("   1) 容量：主干是否为 4 级下采样、参数量是否够（v3 应为 8.28 M）")
    print("   2) 损失权重：w_rec / w_recon_low 是否被正则项盖住")
    print("   3) 训练预算：迭代数是否够（先看 best 出现在第几 epoch）")
else:
    print("→ 训练集拟合尚可但绝对水平低，问题在结构与损失设计，而非训练量。")

if gain < 2.0:
    print("→ 另外：比恒等基线只高不到 2 dB，网络实际学到的很有限。")

if te["psnr"] - best_p < 1.0:
    print("→ 重要：模型只比『单一全局增益 a·I』高不到 1 dB，")
    print("  说明网络几乎没有做空间自适应增强 —— 复杂结构没换来收益。")

if te["illum_std"] < 0.02:
    print("→ 警告：光照图 L 的空间 std ≈ 0，光照分支塌缩成常数图。")
    print("   解耦叙事不成立了 —— plus_illum 与 baseline_unet 的消融会完全重合。")
    print("   先检查 w_noise_l1 是否开了（N 会把空间变化全吸走）、w_illum_smooth 是否过大。")

if te["residual"] < 0.05:
    print("→ 警告：|delta| 太小，主干没在真正修改图像（v3 应 >0.1）。")
