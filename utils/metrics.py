"""PSNR / SSIM / LPIPS 评测工具。

================================================================================
【重要】关于计算约定的说明

PSNR 与 SSIM 的数值强依赖于计算约定，同一个模型在不同约定下可能相差 0.5~1.5 dB。
你在论文里的数字必须与所对比的基线论文使用「同一约定」，否则会被审稿人质疑
"比较不公平"。两种最常见的约定：

  约定 A —— RGB 三通道直接计算
       skimage 默认行为，部分低照度增强论文采用。

  约定 B —— 先转 YCbCr，只取 Y（亮度）通道计算
       BasicSR / Matlab rgb2ycbcr 风格，超分与通用复原领域更常见。

本模块两种都支持，通过 y_channel 参数切换。

【建议流程】
  1. 先跑通 Retinexformer 官方评测脚本，确认它用的是哪种约定；
  2. 在本项目全程统一使用同一约定；
  3. 在论文 "Implementation Details" 中明确写出这一句。
================================================================================

命令行用法：
    python utils/metrics.py --pred results/INDNet/LOLv1 --gt data/LOLdataset/eval15/high
    python utils/metrics.py --pred results/XXX --gt data/LOLdataset/eval15/high --y-channel
    python utils/metrics.py --pred results/XXX --gt data/LOLdataset/eval15/high --lpips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# 基础指标
# --------------------------------------------------------------------------- #
def rgb2ycbcr_y(img: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> Y 通道 [0,1]，BT.601 系数，与 Matlab rgb2ycbcr 对齐。

    输入/输出均为 float64，范围 [0,1]。
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expect HWC RGB image, got shape {img.shape}")
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return np.clip(y, 0.0, 1.0)


def calculate_psnr(
    img1: np.ndarray,
    img2: np.ndarray,
    y_channel: bool = False,
    crop_border: int = 0,
) -> float:
    """计算 PSNR（dB）。输入为 float 数组，范围 [0,1]，shape = (H, W, 3)。"""
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    if img1.shape != img2.shape:
        raise ValueError(f"shape mismatch: {img1.shape} vs {img2.shape}")

    if crop_border > 0:
        c = crop_border
        img1 = img1[c:-c, c:-c]
        img2 = img2[c:-c, c:-c]

    if y_channel:
        img1, img2 = rgb2ycbcr_y(img1), rgb2ycbcr_y(img2)

    mse = float(np.mean((img1 - img2) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def calculate_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    y_channel: bool = False,
    crop_border: int = 0,
) -> float:
    """计算 SSIM。输入为 float 数组，范围 [0,1]。"""
    from skimage.metrics import structural_similarity

    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    if crop_border > 0:
        c = crop_border
        img1 = img1[c:-c, c:-c]
        img2 = img2[c:-c, c:-c]

    if y_channel:
        return float(
            structural_similarity(
                rgb2ycbcr_y(img1), rgb2ycbcr_y(img2), data_range=1.0
            )
        )
    return float(
        structural_similarity(img1, img2, channel_axis=2, data_range=1.0)
    )


class LPIPSMetric:
    """LPIPS 感知相似度（越低越好）。

    依赖 `lpips` 包，首次运行会下载 AlexNet 权重（约 233 MB）。
    """

    def __init__(self, net: str = "alex", device: str | None = None) -> None:
        import torch
        import lpips

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device).eval()
        self.torch = torch

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> float:
        with self.torch.no_grad():
            t1 = self._to_tensor(img1)
            t2 = self._to_tensor(img2)
            return float(self.model(t1, t2).item())

    def _to_tensor(self, img: np.ndarray):
        """[0,1] HWC -> [-1,1] NCHW。"""
        t = self.torch.from_numpy(np.asarray(img, dtype=np.float32))
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = t * 2.0 - 1.0
        return t.to(self.device)


# --------------------------------------------------------------------------- #
# 目录级评测
# --------------------------------------------------------------------------- #
def load_image(path: Path) -> np.ndarray:
    """读取图像为 float32 [0,1] HWC RGB。"""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def find_images(folder: Path) -> dict[str, Path]:
    """收集目录下的图像，返回 {文件名(不含扩展名): 路径}。"""
    if not folder.is_dir():
        raise NotADirectoryError(f"folder not found: {folder}")
    out: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            out[p.stem] = p
    return out


def evaluate_folder(
    pred_dir: Path,
    gt_dir: Path,
    y_channel: bool = False,
    crop_border: int = 0,
    use_lpips: bool = False,
    verbose: bool = True,
) -> dict[str, float]:
    """对两个目录做逐图配对评测，返回平均指标。

    配对规则：按文件名（不含扩展名）一一对应。
    """
    preds = find_images(pred_dir)
    gts = find_images(gt_dir)
    if not preds:
        raise ValueError(f"no images found in pred dir: {pred_dir}")
    if not gts:
        raise ValueError(f"no images found in gt dir: {gt_dir}")

    common = sorted(set(preds) & set(gts))
    if not common:
        raise ValueError(
            "no matched filenames between pred and gt.\n"
            f"  pred sample: {list(preds)[:3]}\n"
            f"  gt   sample: {list(gts)[:3]}\n"
            "检查两侧文件名是否一致（例如 pred 保存时是否加了后缀）。"
        )
    if len(common) != len(preds) or len(common) != len(gts):
        print(
            f"[warn] 文件名不完全匹配：pred {len(preds)} 张 / gt {len(gts)} 张 / "
            f"实际配对 {len(common)} 张",
            file=sys.stderr,
        )

    lpips_metric = LPIPSMetric() if use_lpips else None

    psnr_list, ssim_list, lpips_list = [], [], []
    for name in common:
        pred = load_image(preds[name])
        gt = load_image(gts[name])

        if pred.shape != gt.shape:
            print(
                f"[warn] 尺寸不一致，跳过 {name}: {pred.shape} vs {gt.shape}",
                file=sys.stderr,
            )
            continue

        psnr_list.append(calculate_psnr(pred, gt, y_channel, crop_border))
        ssim_list.append(calculate_ssim(pred, gt, y_channel, crop_border))
        if lpips_metric is not None:
            lpips_list.append(lpips_metric(pred, gt))
        if verbose:
            print(f"  {name:<28} PSNR {psnr_list[-1]:.4f}  SSIM {ssim_list[-1]:.4f}")

    result = {
        "n": len(psnr_list),
        "psnr": float(np.mean(psnr_list)) if psnr_list else float("nan"),
        "ssim": float(np.mean(ssim_list)) if ssim_list else float("nan"),
    }
    if lpips_list:
        result["lpips"] = float(np.mean(lpips_list))

    conv = "Y-channel (约定B)" if y_channel else "RGB (约定A)"
    print("\n" + "=" * 62)
    print(f"  评测目录 : {pred_dir}")
    print(f"  参考目录 : {gt_dir}")
    print(f"  图像数量 : {result['n']}")
    print(f"  计算约定 : {conv}")
    print("-" * 62)
    print(f"  PSNR  : {result['psnr']:.4f} dB")
    print(f"  SSIM  : {result['ssim']:.4f}")
    if "lpips" in result:
        print(f"  LPIPS : {result['lpips']:.4f}")
    print("=" * 62)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="低照度增强结果评测（PSNR / SSIM / LPIPS）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pred", required=True, type=Path, help="增强结果目录")
    parser.add_argument("--gt", required=True, type=Path, help="参考真值目录")
    parser.add_argument(
        "--y-channel",
        action="store_true",
        help="只计算 Y 通道（约定B）；默认 RGB 三通道（约定A）",
    )
    parser.add_argument(
        "--crop-border", type=int, default=0, help="评测前裁掉的边界像素数"
    )
    parser.add_argument("--lpips", action="store_true", help="同时计算 LPIPS")
    parser.add_argument("--quiet", action="store_true", help="只输出汇总")
    args = parser.parse_args()

    evaluate_folder(
        pred_dir=args.pred,
        gt_dir=args.gt,
        y_channel=args.y_channel,
        crop_border=args.crop_border,
        use_lpips=args.lpips,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
