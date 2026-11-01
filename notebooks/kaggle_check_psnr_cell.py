"""训练结果自查：best PSNR 在哪、这次到底训没训。

粘进 Kaggle 一个单元格直接跑。**不需要知道产物在哪一层** ——
同时扫 /kaggle/working（本次会话）和 /kaggle/input（挂载的版本 Output），
按路径长度排序，短的优先。

看三件事：
  1. train.log 存不存在 —— 只有 train.py 被真正调用才会生成它
  2. 「训练计划」里是 [训练] 还是 [跳过] —— 区分"真跑了"和"被幂等逻辑跳过"
  3. best.pth 里的 best_psnr + 配方指纹 —— 最可信，不依赖日志是否完整
"""

import glob
import torch


def _find(pattern: str) -> list[str]:
    """先在 /kaggle/working 找，再找 /kaggle/input；按路径长度升序（浅的优先）。"""
    hits = glob.glob(f"/kaggle/working/**/{pattern}", recursive=True)
    hits += glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return sorted(set(hits), key=len)


def _dump(title: str, paths: list[str]) -> None:
    print(f"\n=== {title} ===")
    if not paths:
        print("   无")
        return
    for p in paths:
        print(f"   {p}")


logs = _find("train.log")
cks = _find("best.pth")
_dump("train.log（有它 = 真的训练了）", logs)
_dump("best.pth", cks)

# ---------------- 日志 ----------------
for p in logs:
    print("\n" + "-" * 72)
    print(f"日志：{p}")
    with open(p, encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip() for ln in f]

    plan = [ln for ln in lines if "[训练]" in ln or "[跳过]" in ln]
    best = [ln for ln in lines if "最佳 PSNR" in ln]

    print("  训练计划：")
    if plan:
        for ln in plan:
            print(f"    {ln.strip()}")
    else:
        print("    （没有训练计划段 —— 说明这个 log 来自旧版脚本）")

    print("  最佳记录（最后 5 条）：")
    if best:
        for ln in best[-5:]:
            print(f"    {ln.strip()}")
    else:
        print("    （还没有 —— 训练刚开头，或中途失败）")

# ---------------- 权重 ----------------
for p in cks:
    print("\n" + "-" * 72)
    print(f"权重：{p}")
    try:
        try:
            rec = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.0 没有 weights_only 参数
            rec = torch.load(p, map_location="cpu")
        cfg = rec.get("cfg") or {}
        tr = cfg.get("train") or {}
        da = cfg.get("data") or {}
        print(f"  best_psnr = {rec.get('best_psnr')}    epoch = {rec.get('epoch')}")
        print(
            "  配方 total_iter / epochs / patch / batch = "
            f"({tr.get('total_iter', 0)}, {tr.get('epochs', 0)}, "
            f"{da.get('patch_size', 0)}, {da.get('batch_size', 0)})"
        )
        print("  ↑ 若是 (0, 200, 256, 16) → 旧配方（6000 iter，18.3 dB 那批）")
        print("    若是 (60000, 200, 128, 8) → 本次探测配方，才是有效结果")
    except Exception as exc:  # 文件损坏 / 不是 checkpoint
        print(f"  读取失败：{exc}")

# ---------------- 结论 ----------------
print("\n" + "=" * 72)
if not logs and not cks:
    print("结论：这个会话里什么都没有 —— 要么没跑过 pipeline，要么跑的是 Run All")
    print("      且会话已被回收（只有 Save & Run All 才会留下产物）。")
elif logs and not any("最佳 PSNR" in open(p, encoding="utf-8", errors="ignore").read() for p in logs):
    print("结论：训练启动了但还没跑完（日志里没有「训练结束」）。等它结束再看。")
else:
    print("结论：训练已完成，上面的 best_psnr 就是本次结果。")
