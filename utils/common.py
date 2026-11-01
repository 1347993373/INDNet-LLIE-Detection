"""通用工具：配置加载、随机种子、日志、图像保存。"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    """读取 YAML 配置。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"empty config: {path}")
    return cfg


def to_namespace(cfg: dict):
    """把嵌套 dict 转成可点访问的对象（只读用途）。"""

    class _NS(dict):
        __getattr__ = dict.__getitem__

        def __setattr__(self, k, v):
            self[k] = v

    def convert(d):
        if isinstance(d, dict):
            return _NS({k: convert(v) for k, v in d.items()})
        if isinstance(d, list):
            return [convert(v) for v in d]
        return d

    return convert(cfg)


# --------------------------------------------------------------------------- #
# 随机种子
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 3407, deterministic: bool = False) -> None:
    """固定所有随机源，保证可复现。

    注意：deterministic=True 会让部分 CUDA 算子回退到非最优实现，训练变慢，
    但结果完全可复现。建议最终出论文数字时开启。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # 允许 TF32（Ampere 及以上提速明显，几乎不影响精度）
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 的种子固定。"""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
def get_logger(name: str = "INDNet", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------- #
# 指标累计
# --------------------------------------------------------------------------- #
class AverageMeter:
    """滑动平均统计量。"""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.val = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n
        self.val = self.sum / max(self.count, 1)

    def __str__(self) -> str:
        return f"{self.name}={self.val:.4f}" if self.name else f"{self.val:.4f}"


# --------------------------------------------------------------------------- #
# 图像保存
# --------------------------------------------------------------------------- #
def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    """把 (3,H,W) 或 (1,H,W)、范围 [0,1] 的 tensor 存成图片。"""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if t.dim() == 4:
        t = t[0]
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)

    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def count_parameters(model: torch.nn.Module) -> int:
    """可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_git_hash() -> str:
    """记录当前代码版本，方便复现。"""
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
