"""过拟合探测：这个架构到底能不能拟合数据。

做法：只用测试集那 15 张图当训练集，从零训 600 步，看训练 PSNR 能冲到多高。
  · 能冲到 28+  ⇒ 表达能力没问题，18.4 是优化/泛化层面的问题
  · 卡在 20 附近 ⇒ 这套输出参数化（R=(I−N)/L + 残差）本身有天花板，必须换结构

粘进 Kaggle 一格直接跑（GPU 约 2–4 分钟）。也可本地跑（在 INDNet/ 根目录下）。
环境变量：PROBE_CKPT / PROBE_DATA / PROBE_ITERS
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

ITERS = int(os.environ.get("PROBE_ITERS", "600"))
PATCH = 128
BATCH = 4
LR = 2e-4


def has_code(p: Path) -> bool:
    try:
        return (p / "train.py").is_file() and (p / "models" / "indnet.py").is_file()
    except OSError:
        return False


root = Path.cwd() if has_code(Path.cwd()) else None
if root is None:
    cands = [Path(p).parent.parent for p in glob.glob("/kaggle/input/**/models/indnet.py", recursive=True)]
    cands = [x for x in cands if has_code(x)]
    cands.sort(key=lambda x: ("INDNet" in str(x), len(str(x))))
    root = cands[0]
sys.path.insert(0, str(root))
os.chdir(root)
print("代码根:", root)

# 只用 checkpoint 里的 cfg 来还原结构（不加载它的权重，从零训练）
cks = glob.glob("/kaggle/working/**/best.pth", recursive=True) + glob.glob("/kaggle/input/**/best.pth", recursive=True)
cks = sorted(set(cks), key=lambda p: (0 if "full_indnet" in p else 1, len(p)))
ck = os.environ.get("PROBE_CKPT") or (cks[0] if cks else None)
rec = torch.load(ck, map_location="cpu", weights_only=False)
cfg = rec["cfg"]
if os.environ.get("PROBE_DATA"):
    cfg["data"]["root"] = os.environ["PROBE_DATA"]

m = cfg["model"]
print(f"结构: base_channels={m.get('base_channels')} num_blocks={m.get('num_blocks')} "
      f"illum={m.get('illumination_branch')} noise={m.get('noise_branch')}")

from datasets.lol_dataset import PairedLowLightDataset, resolve_pair_dirs  # noqa: E402
from models.indnet import build_model  # noqa: E402
from losses.losses import INDNetLoss  # noqa: E402
from utils.metrics import calculate_psnr  # noqa: E402

R0 = Path(cfg["data"]["root"])
if not ((R0 / "our485").is_dir() or (R0 / "eval15").is_dir()):
    hits = sorted({Path(p).parent.parent for p in glob.glob("/kaggle/input/**/our485/low", recursive=True)
                   if (Path(p).parent.parent / "eval15").is_dir()}, key=lambda p: len(str(p)))
    R0 = hits[0]
print("数据根:", R0)

tl, th = resolve_pair_dirs(R0, "train")
el, eh = resolve_pair_dirs(R0, "test")
tiny = PairedLowLightDataset(el, eh, patch_size=PATCH, augment=True)   # 只取 15 张
full = PairedLowLightDataset(el, eh, patch_size=None, augment=False)   # 评估用完整图
print(f"过拟合集 {len(tiny)} 张")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(dev)
print("参数量 %.2f M | 设备 %s" % (sum(p.numel() for p in model.parameters()) / 1e6, dev))

crit = INDNetLoss(cfg)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)


@torch.no_grad()
def eval_full() -> float:
    model.eval()
    ps = []
    for i in range(len(full)):
        it = full[i]
        low = it["low"][None].to(dev)
        high = it["high"][None].to(dev)
        out = model(low)["enhanced"].clamp(0, 1)
        ps.append(calculate_psnr(out[0].permute(1, 2, 0).cpu().numpy(),
                                high[0].permute(1, 2, 0).cpu().numpy()))
    model.train()
    return float(np.mean(ps))


print("\n开始过拟合训练（只 15 张图，看能记住多少）")
model.train()
for it in range(1, ITERS + 1):
    idx = np.random.randint(0, len(tiny), size=BATCH)
    lows, highs = [], []
    for j in idx:
        s = tiny[int(j)]
        lows.append(s["low"])
        highs.append(s["high"])
    low = torch.stack(lows).to(dev)
    high = torch.stack(highs).to(dev)

    out = model(low)
    loss, _ = crit(out, low, high)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if it % 150 == 0 or it == 1:
        p = eval_full()
        head = float(out["delta"].abs().mean().detach()) if "delta" in out else \
            float((out["enhanced"] - out["reflectance"]).abs().mean().detach())
        print(f"  iter {it:4d} | loss {loss.item():.4f} | 训练集 PSNR {p:.2f} | "
              f"delta|mean| {head:.4f} | L均值 {out['illum'].mean():.3f}")

final = eval_full()
print("\n" + "=" * 62)
print(f"  {len(tiny)} 张图的过拟合 PSNR: {final:.2f} dB")
print("=" * 62)
if final >= 28:
    print("→ 表达能力没问题。18.4 是优化/泛化层面的事，加容量或调优化会有收益。")
elif final >= 22:
    print("→ 能记住一部分，但不满。结构偏弱，加容量/加感受野有机会上 22+。")
else:
    print("→ 连这一点图都记不住 ⇒ 输出参数化（除法 + 残差）本身有天花板。")
    print("   必须换结构：改成「直接输出 + 解耦辅助头」。")
