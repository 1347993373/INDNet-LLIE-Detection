"""批量跑消融实验，并汇总成一张可直接填进论文的表。

用法（Kaggle / 本地通用）：
    python scripts/run_ablation.py --base configs/lol_v1_kaggle.yaml \
                                   --out-root /kaggle/working/ablations

    # 只跑其中几个
    python scripts/run_ablation.py --base ... --variants baseline_unet full_indnet

    # 快速试跑（每个只训 5 轮，验证流程，不出正式数字）
    python scripts/run_ablation.py --base ... --epochs 5

    # 已经训完一部分，只看汇总
    python scripts/run_ablation.py --base ... --summary-only

特性：
  * **幂等**：已存在 best.pth 的变体会自动跳过，会话中断后重跑不会从头再来
  * 每个变体有独立 output_dir，互不干扰
  * 结束后打印 Markdown 表格，可直接粘进论文
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 每个变体 = (目录名, 中文标签, 配置覆盖项)
# 前三项是「结构消融」（对应论文 Table 2 的核心部分）
# 后两项是「损失消融」，都建立在完整结构之上
VARIANTS: list[tuple[str, str, dict]] = [
    (
        "baseline_unet",
        "Baseline (U-Net)",
        {"model.illumination_branch": False, "model.noise_branch": False},
    ),
    (
        "plus_illum",
        "+ 光照分支",
        {"model.illumination_branch": True, "model.noise_branch": False},
    ),
    (
        "full_indnet",
        "INDNet (完整)",
        {"model.illumination_branch": True, "model.noise_branch": True},
    ),
    (
        "no_illum_loss",
        "－ 光照平滑损失",
        {
            "model.illumination_branch": True,
            "model.noise_branch": True,
            "loss.w_illum_smooth": 0.0,
        },
    ),
    (
        "no_noise_loss",
        "－ 噪声一致性损失",
        {
            "model.illumination_branch": True,
            "model.noise_branch": True,
            "loss.w_noise_consistency": 0.0,
        },
    ),
]


def set_nested(cfg: dict, dotted_key: str, value) -> None:
    """按 'a.b.c' 形式写入嵌套字典。"""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_best_psnr(ckpt_path: Path) -> tuple[float, int] | None:
    """从 checkpoint 里读回 best_psnr 与实际训练轮数。"""
    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu")
        return float(ckpt.get("best_psnr", float("nan"))), int(ckpt.get("epoch", -1))
    except Exception as exc:
        print(f"  [warn] 无法读取 {ckpt_path}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="批量消融实验")
    parser.add_argument("--base", required=True, type=Path, help="基础配置文件")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("experiments/ablations"),
        help="所有变体的输出根目录",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="只跑指定的变体名（默认全部）",
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数（快速试跑用）")
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    parser.add_argument("--summary-only", action="store_true", help="只汇总，不训练")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，强制重跑")
    args = parser.parse_args()

    base_path = args.base if args.base.is_absolute() else (PROJECT_ROOT / args.base)
    if not base_path.is_file():
        print(f"[错误] 找不到基础配置：{base_path}")
        return 1

    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    selected = [
        v for v in VARIANTS if args.variants is None or v[0] in set(args.variants)
    ]
    if not selected:
        print(f"[错误] 没有匹配的变体。可选：{[v[0] for v in VARIANTS]}")
        return 1

    print("=" * 78)
    print(f"  消融实验：共 {len(selected)} 个变体")
    print(f"  基础配置：{base_path}")
    print(f"  输出目录：{out_root}")
    print("=" * 78)

    results: list[tuple[str, str, float, int, str]] = []

    for name, label, overrides in selected:
        var_dir = out_root / name
        cfg_path = var_dir / "config.yaml"
        ckpt_path = var_dir / "best.pth"

        # ---- 生成该变体的配置 ----
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # 深拷贝
        for k, v in overrides.items():
            set_nested(cfg, k, v)
        cfg["experiment"]["name"] = f"ablation_{name}"
        cfg["experiment"]["output_dir"] = str(var_dir)
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
            # 快跑时同步缩短验证/保存间隔，否则看不到中间结果
            cfg["train"]["val_every"] = max(1, args.epochs // 2)
            cfg["train"]["save_every"] = max(1, args.epochs)

        var_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        print()
        print("-" * 78)
        print(f"  [{label}]")
        print(f"    光照分支 = {overrides.get('model.illumination_branch', cfg['model'].get('illumination_branch'))}"
              f" | 噪声分支 = {overrides.get('model.noise_branch', cfg['model'].get('noise_branch'))}")
        print(f"    配置 -> {cfg_path}")
        print("-" * 78)

        # ---- 训练（幂等：已有结果就跳过）----
        if args.summary_only:
            print("    （--summary-only，跳过训练）")
        elif ckpt_path.is_file() and not args.force:
            print(f"    已存在 {ckpt_path.name}，跳过训练（用 --force 可强制重跑）")
        else:
            cmd = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(cfg_path)]
            if args.device:
                cmd += ["--device", args.device]
            print(f"    执行：{' '.join(cmd)}")
            ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if ret.returncode != 0:
                print(f"    [失败] 变体 {name} 训练返回码 {ret.returncode}，继续下一个")
                results.append((name, label, float("nan"), -1, "训练失败"))
                continue

        # ---- 读取结果 ----
        rec = load_best_psnr(ckpt_path)
        if rec is None:
            results.append((name, label, float("nan"), -1, "无结果"))
        else:
            psnr, epoch = rec
            results.append((name, label, psnr, epoch, ""))

    # ---- 汇总 ----
    print()
    print("=" * 78)
    print("  消融结果汇总")
    print("=" * 78)
    print()
    print("| 配置 | 光照分支 | 噪声分支 | 最佳 PSNR | 状态 |")
    print("|---|---|---|---|---|")

    for (name, label, psnr, epoch, status) in results:
        overrides = next(v[2] for v in VARIANTS if v[0] == name)
        # 变体覆盖项里没提到的开关，说明沿用基础配置（基础配置默认都是 True）
        illum = "✓" if overrides.get("model.illumination_branch", True) else "✗"
        noise = "✓" if overrides.get("model.noise_branch", True) else "✗"
        psnr_str = f"{psnr:.4f}" if psnr == psnr else "—"
        print(f"| {label} | {illum} | {noise} | {psnr_str} | {status} |")

    print()
    # 顺带把完整字段也存一份，方便后续拼论文表
    summary_path = out_root / "ablation_summary.md"
    lines = ["| 配置 | PSNR | 训练轮数 | 状态 |", "|---|---|---|---|"]
    for (name, label, psnr, epoch, status) in results:
        psnr_str = f"{psnr:.4f}" if psnr == psnr else "—"
        lines.append(f"| {label} | {psnr_str} | {epoch} | {status} |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"汇总已保存 -> {summary_path}")
    print()
    print("怎么看这张表：")
    print("  · Baseline -> +光照分支 的涨幅 = 光照解耦的贡献")
    print("  · +光照分支 -> INDNet    的涨幅 = 噪声分支的贡献（本文核心新意）")
    print("  · 后两行去掉损失项后的跌幅 = 各损失项的必要性")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
