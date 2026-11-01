"""配对式低照度数据集（LOL-v1 / LOL-v2-real / 通用 low-high 结构）。

目录布局支持以下几种（大小写不敏感）：
    LOL-v1 : <root>/our485/{low|Low,high|High}
             <root>/eval15/{low|Low,high|High}
    LOL-v2 : <root>/Train/{Low, Normal}      或  {low, high}
             <root>/Test/{Low, Normal}
    通用   : 任意目录，只要 low_dir / high_dir 指向的两个文件夹里
             存在同名文件（扩展名可不同）。

配对规则：按文件名主干（stem）匹配，因此 low/1.png 与 high/1.png 会自动配成一对。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 各数据集里 low / high 文件夹可能的命名
LOW_ALIASES = ("low", "input", "lowlight", "dark")
HIGH_ALIASES = ("high", "normal", "gt", "groundtruth", "reference", "label")


# --------------------------------------------------------------------------- #
# 目录定位
# --------------------------------------------------------------------------- #
def _find_child(parent: Path, aliases: tuple[str, ...]) -> Path | None:
    """在 parent 下按别名（大小写不敏感）找子目录。"""
    if not parent.is_dir():
        return None
    lowered = {a.lower() for a in aliases}
    for child in sorted(parent.iterdir()):
        if child.is_dir() and child.name.lower() in lowered:
            return child
    return None


def resolve_pair_dirs(root: str | Path, split: str = "train") -> tuple[Path, Path]:
    """根据数据集根目录与 split 自动定位 (low_dir, high_dir)。

    会在 root 下**递归向下搜索**（最多 3 层），只要能找到一个同时含有
    low 类子目录与 high 类子目录的文件夹就返回。

    这样设计是为了兼容各种真实布局，例如：
        <root>/our485/{low,high}                  本地解压的 LOL-v1
        <root>/lol_dataset/our485/{low,high}      Kaggle 挂载后多一层嵌套
        <root>/Train/{Low,Normal}                 LOL-v2

    当存在多个候选时，按文件夹名与 split 关键字的匹配程度排序取最优。
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"data root not found: {root}")

    prefer = (
        ("our485", "train")
        if split == "train"
        else ("eval15", "test")
    )
    # 只用于排序：命中 prefer 的排最前，其次是另一个 split 的关键字（避免取错）
    other = ("eval15", "test") if split == "train" else ("our485", "train")

    def scan(folder: Path, depth: int) -> list[tuple[int, Path, Path]]:
        """返回 (优先级, low_dir, high_dir)，优先级越小越优。"""
        found: list[tuple[int, Path, Path]] = []

        low = _find_child(folder, LOW_ALIASES)
        high = _find_child(folder, HIGH_ALIASES)
        if low is not None and high is not None:
            name = folder.name.lower()
            if name in prefer:
                rank = 0
            elif name in other:
                rank = 3          # 明显是另一个 split，排到最后
            else:
                rank = 1
            found.append((rank, low, high))

        if depth > 0:
            skip = {a.lower() for a in LOW_ALIASES + HIGH_ALIASES}
            for child in sorted(folder.iterdir()):
                if child.is_dir() and child.name.lower() not in skip:
                    found.extend(scan(child, depth - 1))
        return found

    results = scan(root, depth=3)
    if results:
        results.sort(key=lambda x: (x[0], len(str(x[1]))))
        return results[0][1], results[0][2]

    raise FileNotFoundError(
        f"无法在 {root} 下找到 {split} 的 low/high 目录（已递归搜索 3 层）。\n"
        f"期望结构例如： {root}/our485/low  与  {root}/our485/high\n"
        f"请运行 python scripts/check_data.py --root {root} 查看诊断。"
    )


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class PairedLowLightDataset(Dataset):
    """低照度增强配对数据集。

    Args:
        low_dir: 低光照图像目录
        high_dir: 正常光照参考目录
        patch_size: 训练时的随机裁剪尺寸；None 表示用原图（评测用）
        augment: 是否做几何增强（训练集 True，测试集 False）
        aug_cfg: 几何增强开关，如 {"hflip": True, "vflip": True, "rot90": True}
                 ⚠️ 早期版本忽略了这个配置（写死 hflip+vflip），
                 导致配置里写 rot90: true 实际不生效、写 vflip: false 也不生效。
    """

    def __init__(
        self,
        low_dir: str | Path,
        high_dir: str | Path,
        patch_size: int | None = 256,
        augment: bool = True,
        aug_cfg: dict | None = None,
    ) -> None:
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.aug_cfg = {
            "hflip": bool((aug_cfg or {}).get("hflip", True)),
            "vflip": bool((aug_cfg or {}).get("vflip", True)),
            "rot90": bool((aug_cfg or {}).get("rot90", False)),
        }

        lows = {p.stem: p for p in sorted(self.low_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS}
        highs = {p.stem: p for p in sorted(self.high_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS}

        if not lows:
            raise ValueError(f"低光照目录为空：{self.low_dir}")
        if not highs:
            raise ValueError(f"参考目录为空：{self.high_dir}")

        common = sorted(set(lows) & set(highs))
        if not common:
            raise ValueError(
                "两侧文件名无法配对，请检查命名是否一致。\n"
                f"  low 示例 : {list(lows)[:3]}\n"
                f"  high 示例: {list(highs)[:3]}"
            )

        missing_high = set(lows) - set(highs)
        missing_low = set(highs) - set(lows)
        if missing_high or missing_low:
            print(
                f"[warn] 未配对的样本已忽略：缺 high {len(missing_high)} 个 / 缺 low {len(missing_low)} 个"
            )

        self.pairs = [(lows[k], highs[k], k) for k in common]
        self.names = common

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        """读图为 float32 [0,1]，HWC RGB。"""
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        low_path, high_path, name = self.pairs[index]
        low = self._load(low_path)
        high = self._load(high_path)

        # 尺寸兜底：以低光照图为准，参考图不足则回退到最小公共尺寸
        if low.shape != high.shape:
            h = min(low.shape[0], high.shape[0])
            w = min(low.shape[1], high.shape[1])
            low, high = low[:h, :w], high[:h, :w]

        if self.patch_size is not None:
            low, high = self._random_crop(low, high, self.patch_size)

        if self.augment:
            low, high = self._geometric_augment(low, high)

        return {
            "low": torch.from_numpy(low.transpose(2, 0, 1).copy()),
            "high": torch.from_numpy(high.transpose(2, 0, 1).copy()),
            "name": name,
        }

    def _random_crop(
        self, low: np.ndarray, high: np.ndarray, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = low.shape[:2]
        if h < size or w < size:
            # 图比 patch 小：先补边再裁，避免训练直接崩
            pad_h, pad_w = max(0, size - h), max(0, size - w)
            low = np.pad(low, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            high = np.pad(high, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w = low.shape[:2]

        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        return (
            low[top : top + size, left : left + size],
            high[top : top + size, left : left + size],
        )

    def _geometric_augment(
        self, low: np.ndarray, high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """只做几何增强，且 low / high 必须同步变换。

        低照度增强任务不要用色彩抖动 / 亮度抖动——那会破坏 low 与 high 之间的
        光照映射关系，等于在给网络制造矛盾的监督信号。

        hflip / vflip / rot90 三个开关**逐个独立生效**（都取自 self.aug_cfg）。
        """
        if self.aug_cfg["hflip"] and random.random() > 0.5:
            low, high = low[:, ::-1], high[:, ::-1]
        if self.aug_cfg["vflip"] and random.random() > 0.5:
            low, high = low[::-1, :], high[::-1, :]
        if self.aug_cfg["rot90"]:
            k = random.randint(0, 3)              # 0/90/180/270 度
            if k:
                low, high = np.rot90(low, k), np.rot90(high, k)
        return np.ascontiguousarray(low), np.ascontiguousarray(high)


# --------------------------------------------------------------------------- #
# 工厂函数
# --------------------------------------------------------------------------- #
def build_datasets(cfg: dict) -> tuple[PairedLowLightDataset, PairedLowLightDataset]:
    """根据配置构建训练集与测试集。

    优先使用配置里显式写出的路径；若不存在，则按 split 自动探测。
    """
    d = cfg["data"]
    root = Path(d["root"])

    train_low = root / d.get("train_low", "")
    train_high = root / d.get("train_high", "")
    if not (train_low.is_dir() and train_high.is_dir()):
        train_low, train_high = resolve_pair_dirs(root, "train")

    test_low = root / d.get("test_low", "")
    test_high = root / d.get("test_high", "")
    if not (test_low.is_dir() and test_high.is_dir()):
        test_low, test_high = resolve_pair_dirs(root, "test")

    patch_size = d.get("patch_size", 256)
    aug = d.get("augment", {}) or {}

    train_set = PairedLowLightDataset(
        train_low, train_high, patch_size=patch_size, augment=True, aug_cfg=aug
    )
    test_set = PairedLowLightDataset(test_low, test_high, patch_size=None, augment=False)

    return train_set, test_set
