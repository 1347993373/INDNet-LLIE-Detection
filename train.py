"""INDNet 训练脚本。

用法：
    python train.py --config configs/lol_v1.yaml
    python train.py --config configs/lol_v1.yaml --resume experiments/INDNet_LOLv1/last.pth
    python train.py --config configs/lol_v1.yaml --smoke      # 只跑几十步，验证管道是否通

消融实验做法：复制配置、改 model 段开关，用不同 output_dir 分别训练。
参见 docs/EXPERIMENT_PLAN.md 的 Table 2。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import build_datasets
from losses.losses import INDNetLoss
from models.indnet import build_model
from utils.common import (
    AverageMeter,
    count_parameters,
    get_logger,
    load_config,
    save_checkpoint,
    save_tensor_image,
    seed_everything,
    worker_init_fn,
)
from utils.metrics import calculate_psnr, calculate_ssim


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard 是可选依赖，缺失时降级为无操作

    class SummaryWriter:  # type: ignore[no-redef]
        """空实现：没有 tensorboard 也能正常训练，只是不记录曲线。"""

        def __init__(self, *args, **kwargs) -> None:
            print("[warn] 未安装 tensorboard，训练曲线将不被记录（不影响训练本身）。")

        def add_scalar(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None


# --------------------------------------------------------------------------- #
def amp_autocast(use_amp: bool):
    """兼容不同 torch 版本的自动混合精度上下文。

    torch 2.4+ 推荐 torch.amp.autocast("cuda", ...)，
    更早的版本只有 torch.cuda.amp.autocast(...)。云端镜像版本不一，这里做兼容。
    """
    if hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def make_grad_scaler(use_amp: bool):
    """同上，GradScaler 的版本兼容封装。"""
    if hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def build_loaders(cfg: dict, pin_memory: bool = True):
    d = cfg["data"]
    train_set, test_set = build_datasets(cfg)

    common = dict(
        num_workers=int(d.get("num_workers", 4)),
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    train_loader = DataLoader(
        train_set, batch_size=int(d.get("batch_size", 8)), shuffle=True, **common
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **common)

    return train_loader, test_loader, train_set, test_set


def build_optimizer(model: torch.nn.Module, cfg: dict, epochs: int | None = None):
    t = cfg["train"]
    name = str(t.get("optimizer", "adam")).lower()
    lr = float(t.get("lr", 2e-4))
    wd = float(t.get("weight_decay", 1e-4))

    if name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"unsupported optimizer: {name}")

    # epochs 由调用方传入（可能来自 total_iter 的换算），保证余弦周期与实际训练长度一致
    epochs = int(epochs if epochs is not None else t.get("epochs", 200))
    sched = str(t.get("scheduler", "cosine")).lower()
    if sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    elif sched == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 5), gamma=0.5)
    else:
        scheduler = None

    return opt, scheduler


# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, loader, device, logger, save_dir: Path | None = None, tag: str = "val"):
    """在测试集上跑一遍，返回 PSNR / SSIM 均值。"""
    model.eval()
    psnr_meter, ssim_meter = AverageMeter("psnr"), AverageMeter("ssim")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        name = batch["name"][0]

        enhanced = model.enhance(low)

        pred_np = enhanced[0].permute(1, 2, 0).float().cpu().numpy()
        gt_np = high[0].permute(1, 2, 0).float().cpu().numpy()

        psnr_meter.update(calculate_psnr(pred_np, gt_np))
        ssim_meter.update(calculate_ssim(pred_np, gt_np))

        if save_dir is not None:
            save_tensor_image(enhanced[0], save_dir / f"{name}_enhanced.png")
            save_tensor_image(low[0], save_dir / f"{name}_input.png")
            save_tensor_image(high[0], save_dir / f"{name}_gt.png")

    model.train()
    logger.info(f"[{tag}] PSNR {psnr_meter.val:.4f} | SSIM {ssim_meter.val:.4f} | n={psnr_meter.count}")
    return psnr_meter.val, ssim_meter.val


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="INDNet 训练")
    parser.add_argument("--config", required=True, type=Path, help="YAML 配置文件")
    parser.add_argument("--resume", type=Path, default=None, help="从 checkpoint 继续训练")
    parser.add_argument("--device", default=None, help="cuda / cuda:0 / cpu，默认自动选择")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：只跑少量步骤")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- 输出目录与日志 ----
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("INDNet", out_dir / "train.log")
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    seed = int(cfg["experiment"].get("seed", 3407))
    seed_everything(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"配置：{args.config}")
    logger.info(f"设备：{device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    logger.info(f"输出：{out_dir.resolve()}")

    if device.type == "cpu":
        logger.warning("当前在 CPU 上训练，速度极慢，仅可用于管道验证。建议改用 GPU。")

    # ---- 数据 ----
    train_loader, test_loader, train_set, test_set = build_loaders(
        cfg, pin_memory=(device.type == "cuda")
    )
    logger.info(f"训练集 {len(train_set)} 对 | 测试集 {len(test_set)} 对")

    # ---- 模型 ----
    model = build_model(cfg).to(device)
    logger.info(f"模型参数量：{count_parameters(model) / 1e6:.3f} M")
    logger.info(
        f"消融开关：光照分支={cfg['model'].get('illumination_branch', True)} "
        f"噪声分支={cfg['model'].get('noise_branch', True)}"
    )

    criterion = INDNetLoss(cfg).to(device)

    # ---- 训练长度：支持「按总迭代数」与「按 epoch」两种口径 ----
    # 为什么需要这个：Retinexformer 官方配方是 2.5e5 次迭代（batch 8 / patch 128），
    # 而我们早期只用 200 epoch × 30 iter = 6000 次，相差 40 倍以上。
    # 用 total_iter 表达训练预算，才能与文献配方直接对齐、也便于换算成本。
    t = cfg["train"]
    iters_per_epoch = max(1, len(train_loader))
    total_iter = int(t.get("total_iter", 0))
    if total_iter > 0:
        epochs = max(1, -(-total_iter // iters_per_epoch))   # 向上取整
        logger.info(
            f"训练预算：target {total_iter} iter | {iters_per_epoch} iter/epoch"
            f" → 需要 {epochs} epoch（按 epoch 换算后向上取整）"
        )
    else:
        epochs = int(t.get("epochs", 200))
    val_every = int(t.get("val_every", 10))
    save_every = int(t.get("save_every", 50))
    log_every = int(t.get("log_every", 50))
    # 迭代数很大时不要每个 epoch 都打一行汇总，否则日志上千行
    epoch_log_every = int(t.get("epoch_log_every", 0))
    if epoch_log_every <= 0:
        epoch_log_every = max(1, epochs // 60) if total_iter > 0 else 1
    grad_clip = float(t.get("grad_clip", 1.0))
    use_amp = bool(t.get("amp", True)) and device.type == "cuda"

    optimizer, scheduler = build_optimizer(model, cfg, epochs)

    start_epoch = 0
    best_psnr = -1.0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_psnr = float(ckpt.get("best_psnr", -1.0))
        logger.info(f"已从 {args.resume} 恢复，起始 epoch {start_epoch}")
    else:
        # 记录随机初始化的验证指标，作为"起点"
        validate(model, test_loader, device, logger, tag="init")

    scaler = make_grad_scaler(use_amp)
    loss_meter = AverageMeter("loss")
    global_step = start_epoch * max(1, len(train_loader))

    for epoch in range(start_epoch, epochs):
        model.train()
        loss_meter.reset()
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast(use_amp):
                outputs = model(low)
                loss, detail = criterion(outputs, low, high)

            if not torch.isfinite(loss):
                logger.error(f"epoch {epoch} iter {it}: loss 非有限值，跳过该 batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(detail["total"])
            global_step += 1

            if it % log_every == 0:
                writer.add_scalar("train/loss", detail["total"], global_step)
                # 用 .get 容错：损失项名称可能随设计调整而变化
                for k in ("out", "refl", "recon_low", "smooth", "noise"):
                    if k in detail:
                        writer.add_scalar(f"train/{k}", detail[k], global_step)
                logger.info(
                    f"epoch {epoch:5d} [{it:4d}/{len(train_loader):4d}] "
                    f"iter {global_step:7d} loss {loss_meter.val:.4f} "
                    f"(out {detail.get('out', float('nan')):.4f} "
                    f"refl {detail.get('refl', float('nan')):.4f} "
                    f"smooth {detail.get('smooth', float('nan')):.4f} "
                    f"noise {detail.get('noise', float('nan')):.4f})"
                )

            if args.smoke and it >= 20:
                logger.info("冒烟测试完成：数据加载、前向、反向、优化器全部正常。")
                return 0

        if scheduler is not None:
            scheduler.step()
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % epoch_log_every == 0 or epoch == epochs - 1:
            logger.info(
                f"epoch {epoch:5d} 完成 | iter {global_step} | 平均 loss {loss_meter.val:.4f}"
                f" | lr {optimizer.param_groups[0]['lr']:.2e} | 耗时 {time.time() - t0:.1f}s"
            )

        # ---- 验证 ----
        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            psnr, ssim = validate(model, test_loader, device, logger, tag=f"epoch{epoch}")
            writer.add_scalar("val/psnr", psnr, epoch)
            writer.add_scalar("val/ssim", ssim, epoch)

            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_psnr": max(best_psnr, psnr),
                "cfg": cfg,
            }
            if psnr > best_psnr:
                best_psnr = psnr
                save_checkpoint(state, out_dir / "best.pth")
                logger.info(f"  ↑ 新的最佳 PSNR {best_psnr:.4f}，已保存 best.pth")

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_psnr": best_psnr,
                    "cfg": cfg,
                },
                out_dir / f"epoch{epoch}.pth",
            )

    save_checkpoint(
        {
            "epoch": epochs - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "cfg": cfg,
        },
        out_dir / "last.pth",
    )
    logger.info(f"训练结束。最佳 PSNR {best_psnr:.4f}。权重位于 {out_dir.resolve()}")
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
