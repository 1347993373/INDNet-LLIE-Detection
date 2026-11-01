"""ExDark 结构深度探查（v2）—— 连文件一起看，找出真实的目录布局与标注文件。

v1 只按"类别名文件夹"搜索，结果 `Dataset/Dataset` 之后就没有子目录了，
说明这个 Kaggle 版本的布局与官方仓库不同。本版改为：
  · 完整打印指定子树（目录 + 文件 + 每个目录的图像计数）
  · 在所有目录名 / 文件名里搜 12 个类别名（大小写不敏感）
  · 递归统计图像总数、识别 zip 等压缩包
只探查，不训练，找不到就打印现状，不抛异常。
"""

from collections import Counter
from pathlib import Path

IN = Path("/kaggle/input")

CLASSES = [
    "bicycle", "boat", "bottle", "bus", "car", "cat",
    "chair", "cup", "dog", "motorbike", "people", "table",
]
CLASS_SET = set(CLASSES)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ARCHIVE_EXT = {".zip", ".tar", ".gz", ".7z", ".rar"}

MAX_DEPTH = 8
MAX_CHILDREN_SHOWN = 25


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def count_images(folder: Path, limit: int = 20000) -> int:
    """递归统计目录下图像数量（带上限，防止在超大目录上卡住）。"""
    n = 0
    stack = [folder]
    while stack:
        cur = stack.pop()
        try:
            for e in cur.iterdir():
                if e.is_dir():
                    stack.append(e)
                elif e.suffix.lower() in IMAGE_EXT:
                    n += 1
                    if n >= limit:
                        return n
        except (PermissionError, OSError):
            continue
    return n


def walk(folder: Path, prefix: str = "", depth: int = MAX_DEPTH, sink=None):
    """打印目录树，并在 sink 里记录所有条目路径。"""
    if depth < 0:
        return
    try:
        entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except (PermissionError, OSError):
        return

    shown = entries[:MAX_CHILDREN_SHOWN]
    for i, e in enumerate(shown):
        last = i == len(shown) - 1
        branch = "└── " if last else "├── "
        if e.is_dir():
            nimg = count_images(e)
            extra = f"  [{nimg} 张图]" if nimg else ""
            print(f"{prefix}{branch}{e.name}/{extra}")
            if sink is not None:
                sink.append(("dir", e))
            walk(e, prefix + ("    " if last else "│   "), depth - 1, sink)
        else:
            size = e.stat().st_size if e.exists() else 0
            print(f"{prefix}{branch}{e.name}  ({human(size)})")
            if sink is not None:
                sink.append(("file", e))
    if len(entries) > MAX_CHILDREN_SHOWN:
        print(f"{prefix}... 还有 {len(entries) - MAX_CHILDREN_SHOWN} 项")


def find_exdark_mount() -> Path | None:
    """定位 ExDark 挂载根：优先找路径里含 exdark 的目录。"""
    cands = []
    for p in IN.rglob("*"):
        if p.is_dir() and "exdark" in p.name.lower():
            cands.append(p)
        if len(cands) > 40:
            break
    if not cands:
        return None
    # 取路径最短的（最外层）
    return sorted(set(cands), key=lambda p: len(str(p)))[0]


def main() -> None:
    root = find_exdark_mount()
    if root is None:
        print("[!] /kaggle/input 下没找到名字含 exdark 的目录。")
        print("    当前顶层：", [p.name for p in sorted(IN.iterdir())])
        return

    print("=" * 78)
    print(f"  ExDark 挂载根: {root}")
    print("=" * 78)
    print()
    print("  完整目录树（最多 8 层）：")
    print(f"  {root.name}/")
    sink: list = []
    walk(root, prefix="  ", depth=MAX_DEPTH, sink=sink)

    # ---------- 统计 ----------
    print()
    print("=" * 78)
    print("  统计")
    print("=" * 78)

    dirs = [p for k, p in sink if k == "dir"]
    files = [p for k, p in sink if k == "file"]

    print(f"  目录数 {len(dirs)} | 文件数 {len(files)}")

    imgs = [f for f in files if f.suffix.lower() in IMAGE_EXT]
    arch = [f for f in files if f.suffix.lower() in ARCHIVE_EXT]
    txts = [f for f in files if f.suffix.lower() == ".txt"]
    other = [f for f in files if f not in imgs and f not in arch and f not in txts]

    print(f"  图像文件 {len(imgs)} 个" + (f"（抽样：{[f.name for f in imgs[:3]]}）" if imgs else ""))
    print(f"  压缩包   {len(arch)} 个" + (f"  <-- 注意！{[(f.name, human(f.stat().st_size)) for f in arch]}" if arch else ""))
    print(f"  txt 标注 {len(txts)} 个")
    print(f"  其他文件 {len(other)} 个" + (f"（抽样：{[f.name for f in other[:5]]}）" if other else ""))

    # ---------- 类别名搜索 ----------
    print()
    print("=" * 78)
    print("  在【目录名】与【文件名】里搜 12 个类别名")
    print("=" * 78)

    hit_dirs = [p for p in dirs if p.name.lower() in CLASS_SET]
    hit_files = [p for p in files if p.stem.lower() in CLASS_SET or p.name.lower() in CLASS_SET]

    print(f"  目录命中 {len(hit_dirs)}: {sorted(p.name for p in hit_dirs)}")
    print(f"  文件命中 {len(hit_files)}: {sorted(p.name for p in hit_files)}")

    if not hit_dirs and not hit_files:
        # 退一步：模糊包含
        fuzzy_d = sorted({p.name for p in dirs if any(c in p.name.lower() for c in CLASS_SET)})
        fuzzy_f = sorted({p.name for p in files if any(c in p.name.lower() for c in CLASS_SET)})
        print(f"  模糊命中（包含而非相等）目录: {fuzzy_d[:15]}")
        print(f"  模糊命中（包含而非相等）文件: {fuzzy_f[:15]}")

    # ---------- txt 内容抽样 ----------
    if txts:
        print()
        print("=" * 78)
        print("  txt 文件抽样")
        print("=" * 78)
        by_dir = Counter(str(f.parent.relative_to(root)) for f in txts)
        print("  按目录分布（top 6）：")
        for d, c in by_dir.most_common(6):
            print(f"    {c:>5} 个  {d}")

        sizes = sorted(txts, key=lambda f: f.stat().st_size)
        for f in [sizes[0], sizes[len(sizes) // 2], sizes[-1]]:
            print()
            print(f"  ---- {f.relative_to(root)}  ({human(f.stat().st_size)}) ----")
            try:
                for ln in f.read_text(encoding="utf-8", errors="replace").splitlines()[:5]:
                    print("      " + ln[:105])
            except Exception as e:
                print("      读取失败:", e)

    print()
    print("=" * 78)
    print("  ✅ 请把以上全部输出复制给我，我据此写检测评测脚本")
    print("=" * 78)


main()
