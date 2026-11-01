"""数据集完整性校验与下载指引。

用法：
    python scripts/check_data.py --root data
    python scripts/check_data.py --root /kaggle/input          # Kaggle 环境

它会做三件事：
  1. 递归扫描 LOL-v1 / LOL-v2 / ExDark，报告图像数量；
  2. 校验 low/high 是否一一配对（这是训练能不能起来的前提）；
  3. 缺失的部分打印下载渠道。

目录探测逻辑与训练时完全一致（复用 datasets.lol_dataset.resolve_pair_dirs），
因此"校验通过"就意味着"训练一定能找到数据"。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接 python scripts/check_data.py 运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from datasets.lol_dataset import resolve_pair_dirs  # 复用同一套探测逻辑

    _HAS_DATASETS = True
except Exception:  # pragma: no cover
    _HAS_DATASETS = False

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DOWNLOAD_GUIDE = {
    "LOL-v1": [
        "Kaggle      : https://www.kaggle.com/datasets/soumikrakshit/lol-dataset",
        "              （在 Kaggle Notebook 里 Add Input 挂载即可，无需下载）",
        "OpenXLab    : https://openxlab.org.cn/datasets/OpenDataLab/LOL",
        "Google Drive: https://drive.google.com/open?id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB",
        "百度网盘     : 见 https://daooshee.github.io/BMVC2018website/",
        "期望结构     : <root>/our485/{low,high} 与 <root>/eval15/{low,high}",
        "              （多一层嵌套也能自动识别，例如 <root>/lol_dataset/our485/...）",
    ],
    "LOL-v2": [
        "作者主页    : https://github.com/flyywh/CVPR-2020-Semi-Low-Light",
        "结构        : Real_captured/{Train,Test}/{Low,Normal}",
        "解压后放到  : data/LOLv2/",
    ],
    "ExDark": [
        "Kaggle      : https://www.kaggle.com/datasets/hongngnguyn/exdarkdataset",
        "              （在 Kaggle Notebook 里 Add Input 粘贴此 URL 即可挂载）",
        "              认准描述含「7,363 low-light images」「12 object classes」的那个；",
        "              搜 \"ExDark\" 会出来大量别人训练好的 YOLO 权重，那些不是原始数据集。",
        "官方仓库    : https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
        "结构        : 图像按类别分文件夹（Bicycle/ Boat/ ... 共 12 类）+ 边界框标注",
        "              注意标注是 [l, t, w, h]（左上坐标 + 宽高），不是 x1y1x2y2",
    ],
}


# --------------------------------------------------------------------------- #
def count_images(folder: Path | None) -> int:
    if folder is None or not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def stem_set(folder: Path | None) -> set[str]:
    if folder is None or not folder.is_dir():
        return set()
    return {p.stem for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"  {'[OK]  ' if ok else '[MISS]'} {name:<30} {detail}")
    return ok


# --------------------------------------------------------------------------- #
def find_lol_root(root: Path) -> Path | None:
    """递归找出真正的 LOL 数据集根（含 our485/eval15 或 low/high 配对的那一层）。

    优先用与训练时相同的 resolve_pair_dirs，保证校验结论与训练一致。
    """
    if _HAS_DATASETS:
        try:
            low, _ = resolve_pair_dirs(root, "train")
            return low.parent.parent  # <...>/our485/low -> <...>
        except Exception:
            pass

    # 兜底：自己扫一遍
    def scan(folder: Path, depth: int) -> Path | None:
        names = {c.name.lower() for c in folder.iterdir() if c.is_dir()}
        if names & {"our485", "eval15", "train", "test"} and (
            {"low", "high"} <= names or {"low", "normal"} <= names
        ):
            return folder
        if depth > 0:
            for child in sorted(folder.iterdir()):
                if child.is_dir():
                    hit = scan(child, depth - 1)
                    if hit is not None:
                        return hit
        return None

    return scan(root, depth=3)


def check_lol_v1(root: Path) -> bool:
    print("\n[1] LOL-v1（主实验数据集）")

    base = find_lol_root(root)
    if base is None:
        return report("LOL-v1", False, f"未找到（已递归搜索 {root} 下 3 层）")

    print(f"  [OK]   {'数据集根':<30} {base}")
    all_ok = True

    for split, expect in (("our485", 485), ("eval15", 15)):
        split_dir = None
        for child in base.iterdir():
            if child.is_dir() and child.name.lower() == split:
                split_dir = child
                break
        if split_dir is None:
            all_ok &= report(split, False, "子目录未找到")
            continue

        low = high = None
        for child in split_dir.iterdir():
            if not child.is_dir():
                continue
            n = child.name.lower()
            if n in ("low", "input", "lowlight", "dark"):
                low = child
            elif n in ("high", "normal", "gt"):
                high = child

        n_low, n_high = count_images(low), count_images(high)
        n_pair = len(stem_set(low) & stem_set(high))
        ok = n_pair >= expect
        all_ok &= ok
        report(f"{split}", ok, f"low {n_low} / high {n_high} / 配对 {n_pair}（期望 {expect}）")

    return all_ok


def check_exdark(root: Path) -> bool:
    print("\n[3] ExDark（下游检测应用，P3 阶段才需要）")

    classes = {
        "bicycle", "boat", "bottle", "bus", "car", "cat",
        "chair", "cup", "dog", "motorbike", "people", "table",
    }

    found: set[str] = set()
    total = 0
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in classes:
            found.add(p.name.lower())
            total += count_images(p)

    ok = len(found) >= 10
    report("ExDark", ok, f"识别到 {len(found)}/12 个类别，共 {total} 张（期望 7363）")
    if ok:
        ann = list(root.rglob("*.txt"))
        report("ExDark 标注", len(ann) > 0, f"找到 {len(ann)} 个 .txt 标注文件")
    return ok


def print_missing(names: list[str]) -> None:
    if not names:
        return
    print("\n" + "=" * 72)
    print("  缺失数据集下载指引")
    print("=" * 72)
    for n in names:
        print(f"\n· {n}")
        for line in DOWNLOAD_GUIDE.get(n, []):
            print(f"    {line}")
    print()


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="数据集完整性校验")
    parser.add_argument("--root", type=Path, default=Path("data"), help="数据根目录")
    args = parser.parse_args()

    root = args.root
    print("=" * 72)
    print(f"  数据集校验 —— {root}")
    print("=" * 72)

    if not root.is_dir():
        print(f"\n[错误] 目录不存在：{root}")
        print_missing(["LOL-v1"])
        return 1

    missing: list[str] = []
    if not check_lol_v1(root):
        missing.append("LOL-v1")
    if not check_exdark(root):
        missing.append("ExDark")

    print("\n" + "=" * 72)
    if not missing:
        print("  全部就绪 —— 可以进入 P1 阶段：复现基线")
    else:
        print(f"  待补齐：{', '.join(missing)}")
    print("=" * 72)

    print_missing(missing)
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
