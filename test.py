"""INDNet 评测脚本。

用法：
    python test.py --config configs/lol_v1.yaml --ckpt experiments/INDNet_LOLv1/best.pth
    python test.py --config configs/lol_v1.yaml --ckpt ... --save-images --decomp --lpips

它会做三件事：
  1. 在测试集上输出增强结果，并计算 PSNR / SSIM / LPIPS；
  2. 把增强图、输入图、参考图成对保存到 results/ 下（供论文 Fig.3 拼图）；
  3. 用 --decomp 额外导出「光照图 / 反射图 / 噪声图」三张分解图 —— 这是论文 Fig.4 的素材，
     也是证明"解耦真的发生了"的直接证据。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import build_datasets
from models.indnet import build_model
from utils.common import get_logger, load_config, save_tensor_image, seed_everything
from utils.metrics import calculate_psnr, calculate_ssim


def to_gray_image(tensor: torch.Tensor) -> torch.Tensor:
    """单通道张量转成可保存的 3 通道灰度图（内部做 min-max 拉伸）。"""
    t = tensor.detach().float().cpu()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    t = t - t.min()
    denom = t.max().clamp(min=1e-8)
    t = t / denom
    return t.repeat(1, 3, 1, 1)[0]


# --------------------------------------------------------------------------- #
@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="INDNet 评测")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path, help="模型权重路径")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None, help="结果输出目录，默认取配置里的 save_dir")
    parser.add_argument("--save-images", action="store_true", help="保存增强/输入/参考三图")
    parser.add_argument("--decomp", action="store_true", help="额外导出光照图/反射图/噪声图")
    parser.add_argument("--y-channel", action="store_true", help="PSNR/SSIM 只算 Y 通道")
    parser.add_argument("--lpips", action="store_true", help="额外计算 LPIPS")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"].get("seed", 3407)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(args.out or cfg["eval"].get("save_dir", "results/INDNet/LOLv1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("INDNet-test", out_dir / "test.log")
    logger.info(f"设备 {device} | 权重 {args.ckpt}")

    # ---- 数据与模型 ----
    _, test_set = build_datasets(cfg)
    loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"已载入权重（epoch={ckpt.get('epoch')}，best_psnr={ckpt.get('best_psnr')}）")

    lpips_metric = None
    if args.lpips:
        try:
            from utils.metrics import LPIPSMetric

            lpips_metric = LPIPSMetric(device=str(device))
        except Exception as exc:
            logger.warning(f"LPIPS 不可用（{exc}），跳过。")

    # ---- 逐图评测 ----
    psnrs, ssims, lpipss = [], [], []

    for batch in loader:
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        name = batch["name"][0]

        outputs = model(low)
        enhanced = outputs["enhanced"].clamp(0.0, 1.0)

        pred_np = enhanced[0].permute(1, 2, 0).float().cpu().numpy()
        gt_np = high[0].permute(1, 2, 0).float().cpu().numpy()

        p = calculate_psnr(pred_np, gt_np, y_channel=args.y_channel)
        s = calculate_ssim(pred_np, gt_np, y_channel=args.y_channel)
        psnrs.append(p)
        ssims.append(s)

        extra = ""
        if lpips_metric is not None:
            lp = lpips_metric(pred_np, gt_np)
            lpipss.append(lp)
            extra = f" LPIPS {lp:.4f}"

        logger.info(f"  {name:<12} PSNR {p:.4f}  SSIM {s:.4f}{extra}")

        if args.save_images:
            save_tensor_image(enhanced[0], out_dir / f"{name}_enhanced.png")
            save_tensor_image(low[0], out_dir / f"{name}_input.png")
            save_tensor_image(high[0], out_dir / f"{name}_gt.png")

        if args.decomp:
            d = out_dir / "decomp"
            save_tensor_image(to_gray_image(outputs["illum"][0]), d / f"{name}_illum.png")
            save_tensor_image(outputs["reflectance"][0].clamp(0, 1), d / f"{name}_reflectance.png")
            # 噪声图有正有负，取绝对值并归一化才看得见分布
            save_tensor_image(to_gray_image(outputs["noise"][0].abs().mean(0, keepdim=True)), d / f"{name}_noise.png")

    # ---- 汇总 ----
    conv = "Y 通道（约定B）" if args.y_channel else "RGB（约定A）"
    logger.info("=" * 58)
    logger.info(f"  图像数量 : {len(psnrs)}")
    logger.info(f"  计算约定 : {conv}")
    logger.info(f"  PSNR     : {np.mean(psnrs):.4f} dB")
    logger.info(f"  SSIM     : {np.mean(ssims):.4f}")
    if lpipss:
        logger.info(f"  LPIPS    : {np.mean(lpipss):.4f}")
    logger.info("=" * 58)

    if args.save_images or args.decomp:
        logger.info(f"结果已保存到 {out_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
