import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working/INDNet")
OUT = Path("/kaggle/working/ablations")

# 本次网络结构有大改动，必须强制刷新代码，避免用到工作区里的旧副本
FORCE_REFRESH = True


def find_code_root(base: Path, max_depth: int = 3):
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
    return sorted(set(found), key=lambda x: len(str(x)))


def find_lol_root(base: Path, max_depth: int = 5):
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
        return sorted(set(strict), key=lambda x: len(str(x)))[0]
    if loose:
        return sorted({p.parent for p in loose}, key=lambda x: len(str(x)))[0]
    return None


# ================= 1. 刷新代码 =================
roots = find_code_root(IN)
if not roots:
    raise SystemExit("找不到代码包，请确认 Input 已挂载 indnet-code（并已上传新版）")

src_root = roots[0]
if FORCE_REFRESH or not (WORK / "train.py").is_file():
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(src_root, WORK)
    print("已刷新代码 ->", WORK)
else:
    print("代码已就位 ->", WORK)

os.chdir(WORK)

# ================= 2. 确认是新版代码 =================
new_model = (WORK / "models" / "indnet.py").read_text(encoding="utf-8")
new_loss = (WORK / "losses" / "losses.py").read_text(encoding="utf-8")

ok_model = "R = (I − N) / L" in new_model or "physical" in new_model.lower() or "head_n" in new_model
ok_loss = "w_refl" in new_loss
print(f"新版模型代码: {'是' if ok_model else '否 —— 请先更新 indnet-code 数据集！'}")
print(f"新版权重项  : {'是' if ok_loss else '否 —— 请先更新 indnet-code 数据集！'}")
if not (ok_model and ok_loss):
    raise SystemExit(
        "检测到工作区代码不是新版。请到 Kaggle 的 indnet-code 数据集页面点 New Version，"
        "上传最新的 indnet-code.tar.gz，再重新运行本格。"
    )

if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

# ================= 3. 定位数据 =================
lol_root = find_lol_root(IN)
if lol_root is None:
    raise SystemExit("没找到 LOL 数据集，请确认 Input 已挂载")
print("LOL 数据根:", lol_root)

# ================= 3.5 算力自检（CPU 上会把 3 小时变成 3 天）=================
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  显存 "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
else:
    print()
    print("!" * 70)
    print("[!] 没有检测到 GPU —— 训练会慢几十倍。")
    print("    本地实测：CPU 上跑 200 轮约 22 小时，三个变体要 3 天。")
    print("    请到右上角 Settings -> Accelerator 选 GPU T4 x2，")
    print("    切换后会话会重启，重新运行本格即可（数据挂载不会丢）。")
    print("!" * 70)
    print()
    if os.environ.get("ALLOW_CPU_TRAIN", "0") != "1":
        raise SystemExit("已在 CPU 环境下停止。确定要用 CPU 训练时，先设 ALLOW_CPU_TRAIN=1。")

base = yaml.safe_load(Path("configs/lol_v1_kaggle.yaml").read_text(encoding="utf-8"))
base["data"]["root"] = str(lol_root)

# ⚠️ 2026-10-20 删除了一段"硬覆盖"损失权重的代码，务必不要再加回来：
#     base.setdefault("loss", {}).update({"w_rec": 1.0, "w_refl": 0.5, "w_recon_low": 0.0})
#   它来自 v1/v2 时代（那时 recon ≡ I 是恒等式，所以置 0 是对的），但一直没删，
#   结果**v3 之后写在 yaml 里的 w_recon_low: 1.0 被无声地改回 0.0**：
#     · 重构约束 enhanced⊙L + N ≈ I_low 从来没生效过 → L 没有任何理由去贴合输入
#     · w_refl 也被改回 0.5，等于把主监督意外放大成 1.5 倍
#   这是"配置写了但没生效"的典型案例，比崩溃更难发现。
#   → 规则：配置只有 yaml 一个来源，单元格里不允许再覆盖损失权重。

# ================= 3.6 训练配方（可用环境变量覆盖，便于分阶段推进）=================
# 为什么需要覆盖：Retinexformer 官方配方是 2.5e5 次迭代，而早期我们只跑了 6000 次。
# 建议分两步走 —— 先小预算探测趋势，确认有效再跑满：
#     os.environ["TOTAL_ITER"]   = "60000"     # 探测档（约 2 小时）
#     os.environ["ABL_VARIANTS"] = "full_indnet"   # 只训完整模型，省钱
#     …确认 PSNR 明显上升后…
#     os.environ["TOTAL_ITER"]   = "250000"    # 对齐 SOTA 配方（约 6–8 小时/变体）
#     os.environ["ABL_VARIANTS"] = ""          # 三个变体全跑
TOTAL_ITER = int(os.environ.get("TOTAL_ITER", "0") or 0)
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "0") or 0)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "0") or 0)
VARIANTS_FILTER = [v.strip() for v in os.environ.get("ABL_VARIANTS", "").split(",") if v.strip()]

if TOTAL_ITER:
    base["train"]["total_iter"] = TOTAL_ITER
if PATCH_SIZE:
    base["data"]["patch_size"] = PATCH_SIZE
if BATCH_SIZE:
    base["data"]["batch_size"] = BATCH_SIZE

_iters = max(1, 485 // int(base["data"].get("batch_size", 8) or 8))
_epochs = -(-int(base["train"].get("total_iter", 0) or 0) // _iters) or int(
    base["train"].get("epochs", 200)
)
print(f"训练配方: total_iter={base['train'].get('total_iter', 0)} "
      f"({_iters} iter/epoch → 约 {_epochs} epoch) | "
      f"patch={base['data'].get('patch_size')} | batch={base['data'].get('batch_size')} | "
      f"aug={base['data'].get('augment')}")


def recipe_of(c: dict) -> tuple:
    """训练配方指纹。用于判断已有权重是不是用当前配方训出来的。

    2026-10-17 追加结构项：只比训练超参不够 —— 改了 base_channels 之后
    `load_state_dict` 会因尺寸不匹配直接崩，而"配方相同 → 跳过训练"会先一步放行。
    结构变了必须重训，所以把结构也纳入指纹。
    """
    c = c or {}
    t = c.get("train", {}) or {}
    d = c.get("data", {}) or {}
    m = c.get("model", {}) or {}
    lc = c.get("loss", {}) or {}
    return (
        int(t.get("total_iter", 0) or 0),
        int(t.get("epochs", 0) or 0),
        int(d.get("patch_size", 0) or 0),
        int(d.get("batch_size", 0) or 0),
        int(m.get("base_channels", 0) or 0),
        int(m.get("num_blocks", 0) or 0),
        bool(m.get("illumination_branch", True)),
        bool(m.get("noise_branch", True)),
        # 2026-10-19 追加：损失权重也是配方的一部分。
        # 只改 w_noise_l1 / w_illum_smooth 时结构没变，若指纹不含它们，
        # 就会"配方一致 → 跳过训练"，拿旧权重的数字冒充新实验（已踩过一次同类坑）。
        round(float(lc.get("w_rec", 0) or 0), 4),
        round(float(lc.get("w_recon_low", 0) or 0), 4),
        round(float(lc.get("w_illum_smooth", 0) or 0), 4),
        round(float(lc.get("w_noise_l1", 0) or 0), 4),
        round(float(lc.get("w_refl", 0) or 0), 4),
        round(float(lc.get("w_illum_sup", 0) or 0), 4),
    )

# ================= 4. 清掉旧结果（仅在强制重训时）=================
# Kaggle 会话重启会清空 /kaggle/working，但万一还在，就别白跑几小时。
FORCE_RETRAIN = os.environ.get("RETRAIN", "0") == "1"
if FORCE_RETRAIN:
    for name in ("baseline_unet", "plus_illum", "full_indnet"):
        d = OUT / name
        if d.exists():
            shutil.rmtree(d)
            print("已删除旧结果:", d.name)

# ================= 5. 训练变体 =================
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = [
    ("baseline_unet", "残差 U-Net 基线", False, False),
    ("plus_illum", "＋光照分支", True, False),
    ("full_indnet", "INDNet (完整)", True, True),
]
if VARIANTS_FILTER:
    _before = len(VARIANTS)
    VARIANTS = [v for v in VARIANTS if v[0] in VARIANTS_FILTER]
    print(f"只跑指定变体: {[v[0] for v in VARIANTS]}（从 {_before} 个中筛出）")
    if not VARIANTS:
        raise SystemExit(f"ABL_VARIANTS={VARIANTS_FILTER} 没有匹配任何变体")


def load_ckpt(path):
    """兼容新旧 torch：2.6+ 的 weights_only 默认值变了。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ---- 训练计划（显式打印）----
# 为什么要这一段：曾经出现过「改了训练配方、跑了两个多小时、结果还是老数字」
# 而日志里毫无提示的情况 —— pipeline 从旧版本的 Output 恢复了权重，
# 于是判定"权重已齐全"直接跳过训练。现在把每个变体是"训练"还是"跳过"、
# 以及判断依据（配方指纹），在开跑前全部列出来。
_want = recipe_of(base)
print()
print("=" * 70, flush=True)
print("消融 / 训练计划", flush=True)
print("=" * 70, flush=True)
print(f"  本次配方 (iter, epochs, patch, batch, base_ch, blocks, illum, noise, +4 个损失权重) = {_want}", flush=True)
for _n, _label, _, _ in VARIANTS:
    _ck = OUT / _n / "best.pth"
    if not _ck.is_file():
        print(f"  [训练] {_label:<18} 无已有权重", flush=True)
        continue
    try:
        _rp = recipe_of((load_ckpt(_ck).get("cfg") or {}))
    except Exception:
        _rp = None
    if _rp == _want:
        print(f"  [跳过] {_label:<18} 配方一致 {_rp}", flush=True)
    else:
        print(f"  [训练] {_label:<18} 配方不同：{_rp} != {_want}", flush=True)
print("=" * 70, flush=True)


results = []
for name, label, illum, noise in VARIANTS:
    var_dir = OUT / name
    cfg_path = var_dir / "config.yaml"
    ckpt = var_dir / "best.pth"

    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["model"]["illumination_branch"] = illum
    cfg["model"]["noise_branch"] = noise
    cfg["experiment"]["name"] = f"ablation_{name}"
    cfg["experiment"]["output_dir"] = str(var_dir)
    var_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print()
    print("=" * 70, flush=True)
    print(f"[{label}]  光照分支={illum}  噪声分支={noise}", flush=True)
    print("=" * 70, flush=True)

    # ---- 已有权重时先核对训练配方 ----
    # ⚠️ 关键：早期用 6000 iter / patch 256 训出的 best.pth 也在同一个目录里。
    # 如果只判断"文件存在"就跳过，会拿旧配方的结果冒充新实验（而且毫无提示）。
    need_train = True
    if ckpt.is_file() and not FORCE_RETRAIN:
        try:
            old_recipe = recipe_of((load_ckpt(ckpt).get("cfg") or {}))
        except Exception as exc:
            print(f"  读取旧权重的 cfg 失败（{exc}），将重新训练", flush=True)
            old_recipe = None
        if old_recipe == recipe_of(cfg):
            print("已有权重（训练配方一致），跳过训练", flush=True)
            need_train = False
        else:
            print(f"  已有权重但训练配方不同，归档后重训", flush=True)
            print(f"    旧配方 iter/epochs/patch/batch/base_ch/blocks/illum/noise/+4权重 = {old_recipe}", flush=True)
            print(f"    新配方 iter/epochs/patch/batch/base_ch/blocks/illum/noise/+4权重 = {recipe_of(cfg)}", flush=True)
            shutil.move(str(ckpt), str(var_dir / "best_old_recipe.pth"))

    if need_train:
        # -u：无缓冲，保证日志不会因缓冲而丢失（Kaggle 后台跑尤其重要）
        ret = subprocess.run([sys.executable, "-u", "train.py", "--config", str(cfg_path)])
        if ret.returncode != 0:
            print("训练失败，跳过该变体", flush=True)
            results.append((label, None))
            continue

    try:
        rec = load_ckpt(ckpt)
        results.append((label, float(rec.get("best_psnr", float("nan")))))
    except Exception as exc:
        print("读取权重失败:", exc, flush=True)
        results.append((label, None))

print()
print("| 配置 | 最佳 PSNR |")
print("|---|---|")
for label, v in results:
    print(f"| {label} | {v:.4f} |" if v is not None and v == v else f"| {label} | — |")

print()
print("这次要看的核心：")
print("  · 若 完整 > 光照分支 > 基线（且差距 > 0.2 dB），说明两个分支真的起作用了")
print("  · 若仍然基本重合，说明问题不在结构而在别处，需要换思路")
