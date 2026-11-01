"""打包代码，用于上传到 Kaggle Dataset。

用法：
    python scripts/pack_for_kaggle.py              # 生成 indnet-code.tar.gz
    python scripts/pack_for_kaggle.py --list        # 只预览会打包哪些文件
    python scripts/pack_for_kaggle.py --name mycode # 自定义包名

设计原则：**白名单**而不是黑名单。
只打包明确需要的代码，其余一律不进去——避免把数据集、训练权重、虚拟环境
误传上去（那些体积大、Kaggle 上又用不到，白白浪费上传时间和存储）。

打包后的压缩包结构（解压到 /kaggle/input/indnet-code/）：
    indnet-code.tar.gz
    ├── train.py
    ├── test.py
    ├── requirements-kaggle.txt
    ├── configs/
    ├── datasets/
    ├── models/
    ├── losses/
    ├── utils/
    ├── scripts/
    └── docs/
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- 白名单：必须打包 ----
INCLUDE_FILES = [
    "train.py",                  # 训练入口
    "test.py",                   # 评测入口
    "README.md",                 # 项目说明
    "requirements.txt",          # 本地依赖（仅作参考）
    "requirements-kaggle.txt",   # Kaggle 依赖（Notebook 实际使用这个）
]

INCLUDE_DIRS = [
    "configs",     # 配置
    "datasets",    # 数据集定义
    "models",      # 网络结构
    "losses",      # 损失函数
    "utils",       # 工具（含 metrics）
    "scripts",     # 校验与打包脚本
    "docs",        # 文档（体积小，便于在 Kaggle 上查阅）
    "notebooks",   # Notebook 本体
]

# ---- 目录内还需排除的 ----
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "Thumbs.db",
]

# ---- 明确不打包的（仅用于提示，白名单机制本身已保证）----
NEVER_INCLUDE = {
    "data": "数据集放在 Kaggle 上挂载即可，不用传",
    "experiments": "训练权重与日志，体积大且每次重跑都会变",
    "results": "评测输出图像，同样不必上传",
    ".venv": "虚拟环境，Kaggle 用自己的",
    ".git": "版本库，无必要",
    "__pycache__": "Python 字节码缓存",
}


def matched(path: Path) -> bool:
    """判断路径是否命中排除规则。"""
    for part in path.parts:
        if part in EXCLUDE_PATTERNS:
            return True
    for pat in EXCLUDE_PATTERNS:
        if pat.startswith("*") and path.name.endswith(pat[1:]):
            return True
    return False


def collect(root: Path) -> list[Path]:
    """收集需要打包的文件（相对 root 的路径）。"""
    files: list[Path] = []

    for name in INCLUDE_FILES:
        p = root / name
        if p.is_file():
            files.append(Path(name))
        else:
            print(f"  [warn] 缺少文件：{name}")

    for d in INCLUDE_DIRS:
        folder = root / d
        if not folder.is_dir():
            print(f"  [warn] 缺少目录：{d}/")
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_dir() or matched(p):
                continue
            files.append(p.relative_to(root))

    return files


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description="打包代码用于上传 Kaggle")
    parser.add_argument("--list", action="store_true", help="只预览文件清单，不生成压缩包")
    parser.add_argument("--name", default="indnet-code", help="压缩包名（默认 indnet-code）")
    parser.add_argument("--out", type=Path, default=None, help="输出目录（默认项目根目录）")
    args = parser.parse_args()

    root = PROJECT_ROOT
    print("=" * 68)
    print(f"  打包目录：{root}")
    print("=" * 68)

    files = collect(root)

    print(f"\n将打包 {len(files)} 个文件：\n")
    by_dir: dict[str, list[Path]] = {}
    for f in files:
        key = f.parts[0] if len(f.parts) > 1 else "(根目录)"
        by_dir.setdefault(key, []).append(f)

    for key in sorted(by_dir):
        items = by_dir[key]
        print(f"  {key}/  ({len(items)} 个)")
        for f in items[:8]:
            print(f"      {f}")
        if len(items) > 8:
            print(f"      ... 还有 {len(items) - 8} 个")

    total_size = sum((root / f).stat().st_size for f in files)
    print(f"\n  原始体积：{human(total_size)}")

    print("\n明确不打包的内容：")
    for name, reason in NEVER_INCLUDE.items():
        exists = "存在" if (root / name).exists() else "不存在"
        print(f"  - {name:<14} ({exists})  {reason}")

    if args.list:
        print("\n（--list 模式，未生成压缩包）")
        return 0

    out_dir = args.out or root
    out_path = out_dir / f"{args.name}.tar.gz"

    with tarfile.open(out_path, "w:gz") as tar:
        for f in files:
            # arcname 用相对路径，确保解压后是扁平结构，不带多余的外层目录
            tar.add(root / f, arcname=str(f).replace("\\", "/"))

    print()
    print("=" * 68)
    print(f"  已生成：{out_path}")
    print(f"  压缩包体积：{human(out_path.stat().st_size)}")
    print("=" * 68)
    print()
    print("下一步：")
    print("  1. 打开 https://www.kaggle.com/datasets -> New Dataset")
    print(f"  2. 上传 {out_path.name}")
    print(f"  3. 标题填 {args.name}（Notebook 按这个名字找代码）")
    print("  4. 可见性选 Private -> Create")
    print()
    print("  改代码后：回该 Dataset 页面点 New Version 重新上传即可")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
