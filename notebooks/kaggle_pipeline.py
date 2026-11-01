"""一条链路跑完：预检 → 补齐消融权重 → ExDark 检测评测。

粘进 Kaggle 一个单元格即可。设计要点：

  · **先预检再训练** —— 数据没挂上、代码包是旧版、没有 GPU 之类的问题，
    10 秒内就能发现，不会等 3 小时训练完才崩。
  · **权重丢失会自动补训** —— Kaggle 会话重启会清空 /kaggle/working。
  · **每步幂等** —— 中途断了重跑不会白费。
  · **带时间戳的阶段标记** —— 跑完在日志里能一眼看出每段花了多久。
  · 子进程一律用 `-u`（无缓冲），保证日志不会因为缓冲而丢失。

可调环境变量（在 main() 之前设置）：
    os.environ["EXDARK_LIMIT"]    = "1500"   # 检测只跑前 1500 张
    os.environ["RETRAIN"]         = "1"      # 强制重训消融变体
    os.environ["ALLOW_CPU_TRAIN"] = "1"      # 允许 CPU 训练（极慢）
    os.environ["SKIP_PREFLIGHT"]  = "1"      # 跳过预检（不建议）
    os.environ["RUN_FINETUNE"]    = "1"      # 追加「检测器微调评测（协议 A2）」

训练预算相关（2026-10-16 新增；透传给消融格，用来分阶段推进）：
    os.environ["TOTAL_ITER"]      = "60000"        # 总迭代数（默认取配置值）
    os.environ["ABL_VARIANTS"]    = "full_indnet"  # 只训指定变体（省钱探测）
    os.environ["PATCH_SIZE"]      = "128"
    os.environ["BATCH_SIZE"]      = "8"

关于「消融/训练」这一步（2026-10-17 修正）：
    本链路**总是**调用消融格，由消融格内部的「配方指纹」
    （total_iter / epochs / patch_size / batch_size）决定是训练还是跳过。
    之前是"权重齐了就跳过"，结果从旧版本 Output 恢复权重后会**永远不训练**，且没有任何提示。
    想跳过这一步用 `os.environ["SKIP_ABLATION"] = "1"`。

关于 RUN_FINETUNE：
    默认不跑。开启后会在冻结协议之外，再跑一遍「每条 pipeline 各自微调检测器」的
    域适配协议（约 2 小时），用于回答"给检测器足够域适配后，哪种增强最有利"。
    相关参数（FT_EPOCHS / FT_MODES / FT_LIMIT …）见 kaggle_exdark_finetune_cell.py。
"""

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working/INDNet")
ABL = Path("/kaggle/working/ablations")
DETECT_OUT = Path("/kaggle/working/detection")
FINETUNE_OUT = Path("/kaggle/working/finetune")

REQUIRED = ["baseline_unet", "plus_illum", "full_indnet"]
PREFLIGHT_SCRIPT = "kaggle_preflight.py"
ABLATION_SCRIPT = "kaggle_ablation_cell.py"
DETECT_SCRIPT = "kaggle_exdark_detect_cell.py"
FINETUNE_SCRIPT = "kaggle_exdark_finetune_cell.py"

_T0 = time.time()


def stamp(msg: str = "") -> None:
    """带时间戳 + 本阶段已耗时，方便跑完后回看时间线。"""
    el = time.time() - _T0
    print(f"[{datetime.now():%H:%M:%S} +{el / 60:5.1f}min] {msg}", flush=True)


def _prefer_dataset(paths: list[Path]) -> list[Path]:
    """排除从「工作区 Output」挂回来的代码副本。

    /kaggle/working/INDNet 里的代码会随版本一起保存。用户为了取回权重，会把
    那个版本的 Output 挂成 Input —— 于是 Input 里就有两份代码：数据集里的新版，
    和 Output 里的旧版。必须优先用数据集那份，否则可能跑到没有修复的旧脚本。

    区分依据：工作区副本的路径里含大写 `INDNet`，而数据集名为 `indnet-code`（小写）。
    """
    ds = [p for p in paths if "INDNet" not in str(p)]
    return ds or paths


def find_in_input(name: str) -> Path | None:
    """按文件名在 /kaggle/input 下递归查找，不依赖数据集命名与挂载层级。"""
    if not IN.exists():
        return None
    cands = sorted(IN.rglob(name), key=lambda p: len(str(p)))
    cands = _prefer_dataset(cands)
    return cands[0] if cands else None


def restore_weights_from_input() -> list[str]:
    """从挂载的 Input 里找回消融权重，省掉重新训练。

    `Save & Run All` 的产物会随版本保存。下次会话重启、/kaggle/working 被清空后，
    只要把那个版本的 Output 作为 Input 挂回来，这里就能把 best.pth 复制进
    /kaggle/working/ablations/，直接跳过 2.4 小时的训练。

    会把查找过程打印出来 —— 挂载路径千奇百怪，出问题时必须能一眼看到原因。
    """
    if not IN.exists():
        stamp("  [权重恢复] /kaggle/input 不存在")
        return []

    all_ck = sorted(IN.rglob("best.pth"), key=lambda p: len(str(p)))
    stamp(f"  [权重恢复] Input 里共有 {len(all_ck)} 个 best.pth，前几个位置：")
    for c in all_ck[:8]:
        stamp(f"      {c}")
    if not all_ck:
        stamp("      （一个都没有 —— 说明上次那个版本的 Output 没挂上）")

    restored = []
    for name in REQUIRED:
        dst = ABL / name / "best.pth"
        if dst.is_file():
            continue
        for c in all_ck:
            if c.parent.name == name:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(c, dst)
                restored.append(name)
                stamp(f"  [权重恢复] {name} <- {c}")
                break
    return restored


def _recipe_of_checkpoint(path: Path):
    """读出 checkpoint 里记录的训练配方指纹；读不到返回 None。

    用来让"权重是旧配方训的"这件事在日志里显式可见 ——
    否则很容易出现「改了 TOTAL_ITER、跑了两小时、结果还是老数字」且毫无提示的情况。
    """
    if not path.is_file():
        return None
    try:
        import torch
        try:
            rec = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            rec = torch.load(path, map_location="cpu")
        c = rec.get("cfg") or {}
        t = c.get("train") or {}
        d = c.get("data") or {}
        m = c.get("model") or {}
        lc = c.get("loss") or {}
        return (
            int(t.get("total_iter", 0) or 0),
            int(t.get("epochs", 0) or 0),
            int(d.get("patch_size", 0) or 0),
            int(d.get("batch_size", 0) or 0),
            int(m.get("base_channels", 0) or 0),
            int(m.get("num_blocks", 0) or 0),
            bool(m.get("illumination_branch", True)),
            bool(m.get("noise_branch", True)),
            round(float(lc.get("w_rec", 0) or 0), 4),
            round(float(lc.get("w_recon_low", 0) or 0), 4),
            round(float(lc.get("w_illum_smooth", 0) or 0), 4),
            round(float(lc.get("w_noise_l1", 0) or 0), 4),
            round(float(lc.get("w_refl", 0) or 0), 4),
            round(float(lc.get("w_illum_sup", 0) or 0), 4),
        )
    except Exception:
        return None


def run_script(path: Path, extra_env: dict | None = None) -> int:
    """用 -u 启动子进程（无缓冲输出），返回退出码。"""
    print()
    print("=" * 78, flush=True)
    stamp(f"【开始】{path.name}")
    print("=" * 78, flush=True)
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    t = time.time()
    r = subprocess.run([sys.executable, "-u", str(path)], env=env)
    stamp(f"【结束】{path.name}  退出码 {r.returncode}  用时 {(time.time() - t) / 60:.1f} 分钟")
    return r.returncode


def main() -> None:
    import torch

    run_finetune = os.environ.get("RUN_FINETUNE", "0") == "1"

    print("=" * 78)
    print("  INDNet 一键链路：预检 -> 补齐消融权重 -> ExDark 检测评测"
          + (" -> 检测器微调评测(A2)" if run_finetune else ""))
    print("  看到这一行，说明 Notebook 里没有残留的旧单元格")
    print("=" * 78)

    # ---------------- 1. 定位脚本 ----------------
    names = [PREFLIGHT_SCRIPT, ABLATION_SCRIPT, DETECT_SCRIPT]
    if run_finetune:
        names.append(FINETUNE_SCRIPT)

    scripts = {}
    for name in names:
        p = find_in_input(name)
        if p is None:
            print(f"[!] 代码包里找不到 {name}")
            print("    请到 Kaggle -> Datasets -> indnet-code -> New Version")
            print("    上传最新 indnet-code.tar.gz，然后重新挂载该数据集。")
            return
        scripts[name] = p
    stamp(f"代码包 -> {scripts[ABLATION_SCRIPT].parent.parent}")

    n_steps = 3 + (1 if run_finetune else 0)

    # ---------------- 2. 预检（关键：避免白跑 3 小时）----------------
    if os.environ.get("SKIP_PREFLIGHT") == "1":
        stamp("已设置 SKIP_PREFLIGHT=1，跳过预检")
    else:
        stamp(f"步骤 0/{n_steps}：环境预检（约 2 分钟，不训练）")
        rc = run_script(scripts[PREFLIGHT_SCRIPT], {"PREFLIGHT_EXIT_CODE": "1"})
        if rc != 0:
            print()
            print("!" * 78)
            print("[!] 预检未通过 —— 已停止，避免白训 3 小时。")
            print("    请按上面的 [FAIL] 提示处理后重新运行本格。")
            print("!" * 78)
            return
        stamp("预检通过")

    if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU_TRAIN", "0") != "1":
        print("[!] 没有 GPU 且未设 ALLOW_CPU_TRAIN=1 —— 已停止")
        return

    # ---------------- 3. 权重恢复 & 是否真的需要训练 ----------------
    # ⚠️ 这里踩过一个坑（2026-10-17）：restore_weights_from_input() 会把**旧版本**的权重
    # 恢复到 ablations/，如果随后据此判断"权重已齐全"就跳过消融格，那么即使把
    # TOTAL_ITER 从 6000 改成 60000，也永远不会真的训练 —— 而且没有任何提示，
    # 只会看到运行时间"正常"、产物"齐全"（其实全是旧的）。
    # 现在改为：**总是调用消融格**，由它内部的「配方指纹」决定是训练还是跳过。
    # 消融格无事可做时只需 2 秒，所以这样最安全。
    restored = restore_weights_from_input()
    if restored:
        stamp(f"已从 Input 恢复消融权重: {restored}")

    # 把已有权重的训练配方打印出来 —— "配方变了却没训练"这类问题应当一眼可见
    for n in REQUIRED:
        rp = _recipe_of_checkpoint(ABL / n / "best.pth")
        if rp is not None:
            stamp(f"  {n} 现有权重配方 "
                  f"(iter, epochs, patch, batch, base_ch, blocks, illum, noise, "
                  f"w_rec, w_recon, w_smooth, w_noiseL1, w_refl, w_illumSup) = {rp}")

    if os.environ.get("SKIP_ABLATION") == "1":
        stamp(f"步骤 1/{n_steps}：已设 SKIP_ABLATION=1，跳过消融/训练")
    else:
        stamp(f"步骤 1/{n_steps}：消融/训练（配方指纹决定是否真训；"
              f"TOTAL_ITER={os.environ.get('TOTAL_ITER', '未设')} "
              f"ABL_VARIANTS={os.environ.get('ABL_VARIANTS', '未设')}）")
        run_script(scripts[ABLATION_SCRIPT])

    still = [n for n in REQUIRED if not (ABL / n / "best.pth").is_file()]
    if still:
        print(f"[!] 缺少权重: {still} —— 后续环节会少对应对比行，但仍继续执行。")

    # ---------------- 4. 检测评测（冻结协议）----------------
    # SKIP_DETECT=1 用于「只跑协议 A2」的场景：A2 自己会做增强+训练+评测，
    # 再跑一遍冻结协议纯属浪费 1–4 小时的 GPU 额度。
    if os.environ.get("SKIP_DETECT") == "1":
        stamp(f"步骤 2/{n_steps}：已设 SKIP_DETECT=1，跳过冻结协议检测评测")
    else:
        stamp(f"步骤 2/{n_steps}：ExDark 检测评测（约 15 – 25 分钟）")
        run_script(scripts[DETECT_SCRIPT])

    # ---------------- 5. 检测器微调评测（协议 A2，可选）----------------
    if run_finetune:
        stamp(f"步骤 3/{n_steps}：检测器微调评测 A2（约 2 小时，含逐 pipeline 微调）")
        run_script(scripts[FINETUNE_SCRIPT])

    # ---------------- 6. 产物清单 ----------------
    print()
    print("=" * 78)
    stamp(f"步骤 {n_steps}/{n_steps}：产物清单")
    print("=" * 78)
    for p in (DETECT_OUT / "detection_summary.txt",):
        print(f"  {'[OK ]' if p.is_file() else '[缺失]'} {p}")
    if run_finetune:
        p = FINETUNE_OUT / "finetune_summary.txt"
        print(f"  {'[OK ]' if p.is_file() else '[缺失]'} {p}")
    for n in REQUIRED:
        p = ABL / n / "best.pth"
        print(f"  {'[OK ]' if p.is_file() else '[缺失]'} {p}")
    if DETECT_OUT.is_dir():
        print(f"  [OK ] {DETECT_OUT}")
    print()
    print("提示：用 Save Version -> Save & Run All 提交，产物会随版本保存。")
    print("      下次会话重启时，可从历史版本把 weights 挂回来，不必重训。")
    stamp("全部结束")


main()
