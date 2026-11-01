"""运行前预检 —— 约 2 分钟，确认这条 3 小时的链路不会白跑。

**不训练、不评测**，只检查环境：
  1. 有没有 GPU
  2. 三个数据集是否都挂载且结构正确（能区分"没挂"和"挂了但内容为空"）
  3. 代码包是否是最新版（含 pipeline 依赖的三个脚本）
  4. /kaggle/working 是否可写、空间够不够
  5. Internet 是否开着（ultralytics 能不能装上、yolov8n.pt 能不能下）

两种用法：
  · 交互会话里单独粘一格跑 → 只打印结果，不抛异常（便于阅读）
  · 被 kaggle_pipeline.py 以子进程调用时，会带 PREFLIGHT_EXIT_CODE=1，
    此时未通过会返回非零退出码，让 pipeline 及时中止、避免白跑 3 小时
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REQUIRED_SCRIPTS = [
    "kaggle_ablation_cell.py",
    "kaggle_exdark_detect_cell.py",
    "kaggle_pipeline.py",
]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALLOW_CPU = os.environ.get("ALLOW_CPU_TRAIN", "0") == "1"

# 平台侧挂载失败的典型处理建议，多处复用
MOUNT_HINT = (
    "这通常是 Kaggle 平台侧的挂载失败（报错里会出现 ERRORED_MOUNTING_DATASET），\n"
    "不是你的操作问题。处理顺序：\n"
    "  ① 刷新页面 -> 右侧 Input 里删掉该数据集 -> 重新 Add Input 加回来\n"
    "  ② 还失败就点右上角 Restart Session 后重试\n"
    "  ③ 或到该数据集页面点 New Notebook，让 Kaggle 自动挂载"
)

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str, detail: str = "") -> bool:
    results.append((ok, msg))
    print(f"  [{'OK  ' if ok else 'FAIL'}] {msg}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")
    return ok


def count_images(d: Path) -> int:
    try:
        return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
    except OSError:
        return 0


def find_lol_root(base: Path, max_depth: int = 5):
    """按结构特征定位 LOL（同时含 our485 与 eval15）。"""
    strict, loose = [], []

    def scan(folder: Path, depth: int):
        try:
            subs = [c for c in folder.iterdir() if c.is_dir()]
        except (PermissionError, OSError):
            return
        names = {c.name.lower() for c in subs}
        if {"our485", "eval15"} <= names:
            strict.append(folder)
            return
        if "low" in names and ({"high", "normal"} & names):
            loose.append(folder)
            return
        if depth > 0:
            for c in subs:
                scan(c, depth - 1)

    scan(base, max_depth)
    if strict:
        return sorted(set(strict), key=lambda p: len(str(p)))[0]
    if loose:
        return sorted({p.parent for p in loose}, key=lambda p: len(str(p)))[0]
    return None


def find_dir_by_name(base: Path, key: str) -> Path | None:
    """按目录名找候选（用于区分"没挂载"和"挂了但没有内容"）。"""
    hits = [p for p in base.rglob("*") if p.is_dir() and key in p.name.lower()]
    return sorted(hits, key=lambda p: len(str(p)))[0] if hits else None


def find_code_root(base: Path, max_depth: int = 4):
    found = []

    def scan(folder: Path, depth: int):
        try:
            if (folder / "train.py").is_file() and (folder / "models" / "indnet.py").is_file():
                found.append(folder)
                return
        except (PermissionError, OSError):
            return
        if depth > 0:
            for c in folder.iterdir():
                if c.is_dir():
                    scan(c, depth - 1)

    scan(base, max_depth)
    return sorted(set(found), key=lambda p: len(str(p)))


def main() -> int:
    print("=" * 78)
    print("  INDNet 运行前预检（不训练、不评测，约 2 分钟）")
    print("=" * 78)

    # ---------------- 1. GPU ----------------
    print("\n【1】算力")
    try:
        import torch

        print(f"      torch {torch.__version__}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            check(True, f"GPU 可用: {p.name} | {p.total_memory / 1024 ** 3:.1f} GB | "
                        f"CUDA {torch.version.cuda}")
            if torch.cuda.device_count() > 1:
                print(f"      检测到 {torch.cuda.device_count()} 块 GPU（代码只用第 0 块）")
        else:
            check(ALLOW_CPU,
                  "没有 GPU" + ("（已设 ALLOW_CPU_TRAIN=1，按允许处理）" if ALLOW_CPU
                                else " —— 请到 Settings -> Accelerator 选 GPU T4 x2"),
                  "" if ALLOW_CPU else
                  "CPU 上单个变体 200 轮约 22 小时。切换加速器后会话会重启"
                  "（数据挂载不丢），重新运行本格即可")
    except Exception as exc:
        check(False, f"torch 不可用: {exc}")

    # ---------------- 2. 数据集 ----------------
    print("\n【2】数据集挂载")

    lol = find_lol_root(IN)
    if lol:
        n_tr, n_te = count_images(lol / "our485" / "low"), count_images(lol / "eval15" / "low")
        check(n_tr == 485 and n_te == 15,
              f"LOL 数据集: our485 {n_tr} 对 / eval15 {n_te} 对",
              f"路径 {lol}" if (n_tr == 485 and n_te == 15)
              else "数量不符，预期 485 / 15 —— 可能挂载不完整\n" + MOUNT_HINT)
    else:
        cand = find_dir_by_name(IN, "lol")
        check(False, "LOL 数据集没找到",
              f"发现目录 {cand}，但里面没有 our485/eval15，挂载可能不完整\n{MOUNT_HINT}"
              if cand else
              "请挂载 https://www.kaggle.com/datasets/soumikrakshit/lol-dataset")

    exd = find_dir_by_name(IN, "exdark")
    if exd is None:
        check(False, "ExDark 数据集没找到",
              "请挂载 https://www.kaggle.com/datasets/hongngnguyn/exdarkdataset")
    else:
        csvs = list(exd.rglob("*.csv"))
        best, n_img = None, 0
        for d in exd.rglob("*"):
            if d.is_dir():
                c = count_images(d)
                if c > n_img:
                    best, n_img = d, c
        if n_img > 1000 and csvs:
            check(True, f"ExDark: 图像 {n_img} 张 / 标注 {csvs[0].name}",
                  f"路径 {best}")
        else:
            check(False,
                  f"ExDark 挂载不完整（图像 {n_img} 张 / 标注 {len(csvs)} 个）",
                  MOUNT_HINT)

    # ---------------- 3. 代码包 ----------------
    print("\n【3】代码包")
    roots = find_code_root(IN)
    if not roots:
        check(False, "找不到代码包（需同时含 train.py 与 models/indnet.py）",
              "请挂载你自己的 indnet-code 数据集")
    else:
        root = roots[0]
        check(True, "代码包已挂载", f"路径 {root}")
        missing = [n for n in REQUIRED_SCRIPTS if not list(root.rglob(n))]
        check(not missing,
              "pipeline 依赖的脚本齐全" if not missing else f"缺少脚本: {missing}",
              "" if not missing else
              "说明代码包还是旧版本 —— 请到数据集页点 New Version 上传最新压缩包")

    # ---------------- 4. 输出目录 ----------------
    print("\n【4】输出目录")
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        probe = WORK / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        free = shutil.disk_usage(str(WORK)).free / 1024 ** 3
        check(True, f"/kaggle/working 可写，剩余 {free:.1f} GB")
    except Exception as exc:
        check(False, f"/kaggle/working 不可写: {exc}")

    # ---------------- 5. Internet 与依赖 ----------------
    print("\n【5】Internet 与依赖")
    if os.environ.get("PREFLIGHT_SKIP_INSTALL") == "1":
        print("      [跳过] 已设置 PREFLIGHT_SKIP_INSTALL=1")
    else:
        if importlib.util.find_spec("ultralytics") is None:
            print("      正在安装 ultralytics ...")
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "ultralytics"],
                check=False,
            )
            if r.returncode == 0:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                "py-cpuinfo", "psutil", "pyyaml", "requests", "scipy"],
                               check=False)
        ok = importlib.util.find_spec("ultralytics") is not None
        if ok:
            import ultralytics
            check(True, f"ultralytics 可用 {ultralytics.__version__}")
        else:
            check(False, "ultralytics 安装失败",
                  "多半是 Internet 没开：Settings -> Internet -> On")

        if ok:
            try:
                from ultralytics import YOLO

                YOLO("yolov8n.pt")
                check(True, "yolov8n.pt 可获取")
            except Exception as exc:
                check(False, f"yolov8n.pt 获取失败: {exc}", "确认 Internet 已打开")

    # ---------------- 汇总 ----------------
    print()
    print("=" * 78)
    n_fail = sum(1 for ok, _ in results if not ok)
    if n_fail == 0:
        print("  全部通过 —— 可以跑正式流程了")
        print()
        print("  预计耗时（GPU T4 单卡）:")
        print("    消融 3 个变体 · 各 200 轮       约 2.5 – 3.5 小时")
        print("    ExDark 检测 4 种输入 × 7362 张    约 15 – 25 分钟")
        print("    环境安装 / 数据索引 / 汇总        约 5 分钟")
        print("    ------------------------------------------------")
        print("    合计                            约 3 – 4 小时")
    else:
        print(f"  {n_fail} 项未通过 —— 请先解决下列问题，再跑正式流程：")
        for ok, msg in results:
            if not ok:
                print(f"    · {msg}")
    print("=" * 78)
    return n_fail


_n_fail = main()

# 被 pipeline 以子进程调用时，用退出码传递结果（避免在交互格子里抛异常）
if os.environ.get("PREFLIGHT_EXIT_CODE") == "1":
    sys.exit(0 if _n_fail == 0 else 1)
