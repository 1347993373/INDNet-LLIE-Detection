"""ExDark 检测器微调评测（协议 A2）—— 让每种增强都获得域适配的机会。

【为什么需要这个协议】

冻结协议（一个 COCO 预训练检测器直接测原图 / 增强图）下的结果是负的：

    原图 0.5338 | Gamma 0.4981 | CLAHE 0.5348(+0.001) | 各增强变体 -0.06 ~ -0.15

连 CLAHE 都只拿到 +0.001（噪声量级 ±0.005），说明这个协议本身就最容易出负结果：
COCO 预训练检测器在 ExDark 上存在域偏移，而增强又会进一步改变图像分布，
两头不利。文献里报增益的工作基本都做了「域适配」或「检测感知训练」。

【本协议怎么做】

每条 pipeline（原图 / Gamma / CLAHE / 三个消融变体）**各自微调一个检测器**：
在**自己的增强训练集**上训练，在**自己的增强测试集**上测试。
→ 每条 pipeline 都是域内，谁都不吃亏；差别只剩下"哪种前处理更有利于检测"。

【钉死的公平性（代码里写死，不允许逐方法调参）】

  1. 同一份图像划分 —— 按图像切分并写入 split.json，所有 pipeline 共用
  2. 完全相同的超参 —— epochs / lr / imgsz / batch / 增强开关 / 随机种子
  3. 每个检测器只见过自己那种图像（不会偷看别人的）
  4. 关闭 hsv_h/s/v、mosaic、mixup —— 否则检测器自带的颜色增强会掩盖
     "不同增强方法"的差异，而那恰恰是我们要测的东西
  5. 关闭早停（patience >= epochs）—— 保证每种方法拿到相同的训练预算

【另一张表：交叉矩阵】

FT_CROSS_EVAL=1（默认开）时，额外用「原图」微调的检测器去测其它 pipeline 的测试图。
这对应真实的即插即用场景（已有检测器 + 换个前处理）。两张表一起看，才能分清
"增强本身的效果"与"域适配带来的收益"各自有多大。

可调环境变量（写在 main() 调用之前）：
    os.environ["FT_MODES"]    = "original,clahe,full_indnet"  # 只跑这几个（省时）
    os.environ["FT_EPOCHS"]   = "40"     # 微调轮数
    os.environ["FT_LIMIT"]    = "600"    # 每类只取 N 张，冒烟测试用
    os.environ["FT_CROSS_EVAL"] = "0"    # 关掉交叉矩阵
    os.environ["FT_FORCE"]    = "1"      # 强制重做图像与训练
"""

import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# 必须在 import torch 之前设置：缓解显存碎片
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

try:
    import cv2 as _cv2
except Exception:  # pragma: no cover
    _cv2 = None

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working/INDNet")
ABL = Path("/kaggle/working/ablations")
DS_ROOT = Path("/kaggle/working/yolo_ds")
OUT = Path("/kaggle/working/finetune")

SPLIT_JSON = Path("/kaggle/working/exdark_split.json")
SUMMARY_TXT = OUT / "finetune_summary.txt"

CLASS_NAMES = ["bicycle", "boat", "bottle", "bus", "car", "cat",
               "chair", "cup", "dog", "motorbike", "people", "table"]
N_CLS = len(CLASS_NAMES)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "40"))
FT_IMGSZ = int(os.environ.get("FT_IMGSZ", "640"))
FT_BATCH = int(os.environ.get("FT_BATCH", "16"))
FT_SEED = int(os.environ.get("FT_SEED", "3407"))
# 多种子：数据集划分固定用 FT_SEED（隔离"训练随机性"这一个变量），
# 训练重复 FT_SEEDS 次得到 mean ± std。审稿人一定会问单种子的方差。
FT_SEEDS = [int(s) for s in os.environ.get("FT_SEEDS", str(FT_SEED))
            .replace(" ", "").split(",") if s]
FT_LIMIT = int(os.environ.get("FT_LIMIT", "0"))
MAX_SIDE = int(os.environ.get("DATASET_MAX_SIDE", "1024"))
CROSS_EVAL = os.environ.get("FT_CROSS_EVAL", "1") == "1"
ALLOW_CPU = os.environ.get("ALLOW_CPU_TRAIN", "0") == "1"
FORCE = os.environ.get("FT_FORCE", "0") == "1"
# 每个模式训完即删训练/验证图像（保留 test 供交叉评测），用于磁盘紧张时省空间
CLEAN_TRAIN = os.environ.get("FT_CLEAN_TRAIN", "0") == "1"
SPLIT_RATIOS = (0.60, 0.15, 0.25)

_MODE_ORDER = ["original", "gamma", "clahe", "baseline_unet", "plus_illum", "full_indnet"]
_MODEL_MODES = {"baseline_unet", "plus_illum", "full_indnet"}
_LABEL = {
    "original": "原图（未增强）", "gamma": "Gamma γ=0.5（经典）", "clahe": "CLAHE（经典）",
    "baseline_unet": "＋残差 U-Net 基线", "plus_illum": "＋光照分支",
    "full_indnet": "＋INDNet（完整）",
}
_env_modes = [m.strip() for m in os.environ.get("FT_MODES", "").split(",") if m.strip()]
MODES = [m for m in (_env_modes or _MODE_ORDER) if m in _MODE_ORDER]
if not MODES:
    MODES = list(_MODE_ORDER)

_T0 = time.time()


def log(msg: str = "") -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 定位（与 detect 格保持同一套逻辑）
# --------------------------------------------------------------------------- #
def find_code_root(base: Path, max_depth: int = 4):
    found = []

    def scan(folder: Path, depth: int):
        try:
            # 注意：iterdir 必须在 try 内 —— 路径不存在或中途消失时要能安全返回
            if (folder / "train.py").is_file() and (folder / "models" / "indnet.py").is_file():
                found.append(folder)
                return
            if depth <= 0:
                return
            subs = [c for c in folder.iterdir() if c.is_dir()]
        except OSError:
            return
        for c in subs:
            scan(c, depth - 1)

    scan(base, max_depth)
    found = sorted(set(found), key=lambda x: len(str(x)))
    ds = [p for p in found if "INDNet" not in str(p)]   # 排除工作区 Output 里的旧副本
    return ds or found


def find_exdark():
    """返回 (图像目录, 标注 csv)。"""
    mount = None
    for p in IN.rglob("*"):
        if p.is_dir() and "exdark" in p.name.lower():
            mount = p if mount is None or len(str(p)) < len(str(mount)) else mount
    if mount is None:
        return None, None
    csvs = list(mount.rglob("*.csv"))
    ann = max(csvs, key=lambda f: f.stat().st_size) if csvs else None
    best, best_n = None, 0
    for d in mount.rglob("*"):
        if not d.is_dir():
            continue
        try:
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT)
        except OSError:
            continue
        if n > best_n:
            best, best_n = d, n
    return best, ann


def index_images(img_dir: Path):
    """按 stem 建索引。Kaggle 是 Linux 大小写敏感，拼扩展名的写法会漏掉 .JPG。"""
    idx = {}
    try:
        entries = list(img_dir.iterdir())
    except OSError:
        return idx
    for f in entries:
        try:
            if f.is_file() and f.suffix.lower() in IMAGE_EXT:
                idx.setdefault(f.stem, f)
        except OSError:
            continue
    return idx


def restore_weights_from_input():
    """从挂载的 Input 里找回消融权重（会话重启会清空 /kaggle/working）。"""
    if not IN.exists():
        return []
    all_ck = sorted(IN.rglob("best.pth"), key=lambda p: len(str(p)))
    restored = []
    for name in _MODEL_MODES:
        dst = ABL / name / "best.pth"
        if dst.is_file():
            continue
        for c in all_ck:
            if c.parent.name == name:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(c, dst)
                restored.append(name)
                break
    return restored


# --------------------------------------------------------------------------- #
# 标注解析
# --------------------------------------------------------------------------- #
def parse_annotations(csv_path: Path):
    """解析 annotations.csv → ({stem: [(cls, cx, cy, w, h), ...]}, meta)。

    这个 Kaggle 版本（hongngnguyn/exdarkdataset）的结构是「平铺图像 + 单一 CSV」：
        表头  image_path, x_tl, y_tl, x_br, y_br, w, h, class, image_height, image_width
        框    左上角 + 右下角 角点（绝对像素），并冗余给出 w/h 可做交叉校验
        class **0 基编码 0~11**（不是官方文档的 1~12）
    """
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        raise SystemExit(f"[!] {csv_path} 是空文件")

    header = [h.strip().lower() for h in rows[0]]
    body = rows[1:]

    need = ["image_path", "x_tl", "y_tl", "x_br", "y_br", "class", "image_height", "image_width"]
    missing = [k for k in need if k not in header]
    if missing:
        raise SystemExit(f"[!] 标注表头缺少列 {missing}\n    实际表头: {rows[0]}")

    i_img = header.index("image_path")
    i_x1, i_y1 = header.index("x_tl"), header.index("y_tl")
    i_x2, i_y2 = header.index("x_br"), header.index("y_br")
    i_cls = header.index("class")
    i_h, i_w = header.index("image_height"), header.index("image_width")
    i_bw = header.index("w") if "w" in header else None
    i_bh = header.index("h") if "h" in header else None

    # ---- class 编码判定：出现 12 → 1 基；出现 0 而不出现 12 → 0 基 ----
    raw_cls = []
    for r in body:
        try:
            raw_cls.append(int(float(r[i_cls])))
        except Exception:
            pass
    cnt = Counter(raw_cls)
    if 12 in cnt and 0 not in cnt:
        offset = -1
        mode = "数字编码，1 基（1~12）"
    else:
        offset = 0
        mode = "数字编码，0 基（0~11）"
    log(f"  class 编码判定: {mode}；偏移 {offset:+d}")
    bad_cls = [v for v in cnt if not (0 <= v + offset < N_CLS)]
    if bad_cls:
        log(f"  [!] 出现越界编码 {sorted(bad_cls)[:8]} —— 这些行会被丢弃")

    gts: dict[str, list] = defaultdict(list)
    sizes: dict[str, tuple[int, int]] = {}
    unparsed = 0
    bad_class = 0
    incons = []

    for r in body:
        try:
            stem = Path(r[i_img].strip()).stem
            c = int(float(r[i_cls])) + offset
            if not (0 <= c < N_CLS):
                bad_class += 1
                continue
            x1, y1 = float(r[i_x1]), float(r[i_y1])
            x2, y2 = float(r[i_x2]), float(r[i_y2])
            W, H = float(r[i_w]), float(r[i_h])
            if W <= 1 or H <= 1:
                unparsed += 1
                continue
            if i_bw is not None and i_bh is not None:
                try:
                    incons.append(max(abs((x2 - x1) - float(r[i_bw])),
                                      abs((y2 - y1) - float(r[i_bh]))))
                except Exception:
                    pass
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 1 or y2 - y1 < 1:
                unparsed += 1
                continue
            gts[stem].append((c,
                              (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                              (x2 - x1) / W, (y2 - y1) / H))
            sizes[stem] = (int(W), int(H))
        except Exception:
            unparsed += 1

    log(f"  解析 {sum(len(v) for v in gts.values())} 个框 / {len(gts)} 张图"
        f"；越界类别 {bad_class}；解析失败 {unparsed}")
    if incons:
        a = np.array(incons)
        log(f"  角点↔宽高校验: 最大 {a.max():.0f}px 中位 {np.median(a):.0f}px"
            f" 不一致(>2px) {(a > 2).sum()}/{len(a)} = {(a > 2).mean() * 100:.3f}%")
    if len(gts) < 10:
        raise SystemExit("[!] 有效标注少于 10 张图，几乎肯定是解析错误 —— 已停止")

    dist = Counter(c for v in gts.values() for c, *_ in v)
    log("  类别分布: " + ", ".join(f"{CLASS_NAMES[k]}={dist.get(k, 0)}"
                                   for k in range(N_CLS)))
    return gts, sizes


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def make_split(stems: list[str]) -> dict[str, list[str]]:
    """按图像切分并落盘。所有 pipeline 共用这一份，保证完全一致。"""
    if SPLIT_JSON.is_file() and not FORCE:
        saved = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
        if set(saved.get("train", [])) <= set(stems):
            log(f"  复用已有划分 -> {SPLIT_JSON}")
            return saved

    s = sorted(stems)
    random.Random(FT_SEED).shuffle(s)
    n = len(s)
    n_tr = int(n * SPLIT_RATIOS[0])
    n_va = int(n * SPLIT_RATIOS[1])
    split = {"train": s[:n_tr], "val": s[n_tr:n_tr + n_va], "test": s[n_tr + n_va:]}
    SPLIT_JSON.write_text(json.dumps({**split, "_seed": FT_SEED,
                                      "_ratios": SPLIT_RATIOS}, ensure_ascii=False),
                          encoding="utf-8")
    log(f"  已生成划分（seed={FT_SEED}）-> {SPLIT_JSON}")
    return split


# --------------------------------------------------------------------------- #
# 图像准备：先缩放到 MAX_SIDE，再做对应增强
# --------------------------------------------------------------------------- #
def _resize_max(t, max_side: int):
    import torch.nn.functional as F
    h, w = t.shape[-2:]
    side = max(h, w)
    if side <= max_side:
        return t
    sc = max_side / side
    return F.interpolate(t, size=(max(16, int(round(h * sc))), max(16, int(round(w * sc)))),
                         mode="bilinear", align_corners=False, antialias=True)


def _clahe_hwc(rgb_u8: np.ndarray) -> np.ndarray:
    lab = _cv2.cvtColor(rgb_u8, _cv2.COLOR_RGB2LAB)
    cl = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return _cv2.cvtColor(lab, _cv2.COLOR_LAB2RGB)


def prepare_dataset(mode: str, split: dict, img_map: dict, gts: dict,
                    net, dev) -> Path:
    """把某个 mode 的 train/val/test 图像写到 DS_ROOT/<mode>/ 下，并生成 data.yaml。

    图像统一先缩放到 MAX_SIDE（长边），再做增强 —— 这样：
      · 所有 pipeline 走同一条尺寸路径，比较公平
      · 增强永远在 <=1024 下进行，天然不会 OOM（ExDark 里有 4301px 的图）
      · YOLO 反正会 letterbox 到 640，信息量不损失

    标签（归一化 cx/cy/w/h）与图像尺寸无关，只要长宽比不变就可通用，
    因此各 mode 的标签内容相同，但必须各存一份（ultralytics 按路径推导 labels 目录）。
    """
    import torch

    root = DS_ROOT / mode
    if FORCE:
        shutil.rmtree(root, ignore_errors=True)
    counts = {}
    for sp in ("train", "val", "test"):
        img_dir = root / "images" / sp
        lbl_dir = root / "labels" / sp
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        stems = split[sp]

        # 先清理过期文件 —— 换了 FT_LIMIT 或划分之后，旧图会残留下来污染训练和评测
        # （实测踩过：20 张的冒烟跑里混进了上一次 60 张划分的图）
        want = {f"{s}.jpg" for s in stems}
        stale = [p for p in img_dir.glob("*.jpg") if p.name not in want]
        if stale:
            for p in stale:
                p.unlink(missing_ok=True)
                (lbl_dir / f"{p.stem}.txt").unlink(missing_ok=True)
            log(f"    [{mode}/{sp}] 清掉 {len(stale)} 个过期文件")

        n_new = 0
        for i, s in enumerate(stems):
            boxes = gts.get(s) or []
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
            (lbl_dir / f"{s}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

            dst = img_dir / f"{s}.jpg"
            if dst.is_file() and not FORCE:
                continue

            from PIL import Image
            arr = np.asarray(Image.open(img_map[s]).convert("RGB"), dtype=np.float32) / 255.0
            t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0)
            del arr
            t = _resize_max(t, MAX_SIDE)

            if mode == "original":
                out = t
            elif mode == "gamma":
                out = t.pow(0.5)
            elif mode == "clahe":
                u8 = (t[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                out = torch.from_numpy(
                    _clahe_hwc(u8).astype(np.float32).transpose(2, 0, 1) / 255.0
                ).unsqueeze(0)
            else:
                with torch.no_grad():
                    out = net.enhance(t.to(dev)).cpu()
                if not torch.isfinite(out).all():
                    out = t
            u8 = ((out[0].permute(1, 2, 0).numpy().clip(0.0, 1.0)) * 255).round().astype(np.uint8)
            Image.fromarray(u8).save(dst, quality=92)
            n_new += 1
            del t, out, u8

        counts[sp] = len(stems)
        n_img = len(list(img_dir.glob("*.jpg")))
        flag = "" if n_img == len(stems) else f"  [!] 实际文件 {n_img} 与划分 {len(stems)} 不符"
        log(f"    [{mode}/{sp}] {len(stems)} 张（新写 {n_new}）{flag}")

    cfg = DS_ROOT / f"{mode}.yaml"
    cfg.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8")
    return cfg


# --------------------------------------------------------------------------- #
# 微调与评测
# --------------------------------------------------------------------------- #
FIXED_HPARAMS = dict(
    # 统一关闭会掩盖"颜色差异"的增强 —— 这恰恰是我们要测的东西
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
    degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5, flipud=0.0,
)


def ensure_ultralytics():
    try:
        import ultralytics  # noqa: F401
        return
    except Exception:
        pass
    log("正在安装 ultralytics ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)


def finetune(mode: str, yaml_path: Path, dev, seed: int) -> Path | None:
    from ultralytics import YOLO

    # 单种子时沿用旧目录名（可复用历史权重）；多种子时按 seed 分目录
    run_name = mode if len(FT_SEEDS) == 1 else f"{mode}_s{seed}"
    weights = OUT / "runs" / run_name / "weights" / "best.pt"
    if weights.is_file() and not FORCE:
        log(f"    [{run_name}] 已有微调权重，跳过训练")
        return weights

    model = YOLO("yolov8n.pt")           # COCO 预训练起点（所有 mode 完全相同）
    model.train(
        data=str(yaml_path), epochs=FT_EPOCHS, imgsz=FT_IMGSZ, batch=FT_BATCH,
        device=dev, project=str(OUT / "runs"), name=run_name, exist_ok=True,
        seed=seed, deterministic=True, workers=2, verbose=False, plots=False,
        patience=FT_EPOCHS,              # 关闭早停 → 各方法训练预算一致
        optimizer="SGD", lr0=0.01, lrf=0.01, warmup_epochs=3.0, cos_lr=False,
        **FIXED_HPARAMS,
    )
    return weights if weights.is_file() else None


def evaluate(weights: Path, yaml_path: Path, dev):
    """在 yaml 对应的 test 划分上评测，返回 (mAP@0.5, 分类型 mAP dict)。"""
    from ultralytics import YOLO

    r = YOLO(str(weights)).val(data=str(yaml_path), split="test", imgsz=FT_IMGSZ,
                               batch=FT_BATCH, device=dev, verbose=False, plots=False)
    per_cls = {}
    try:
        names = r.names if isinstance(r.names, dict) else dict(enumerate(r.names))
        ap = getattr(r.box, "ap50", None)
        if ap is not None:
            for i, v in enumerate(np.asarray(ap).ravel()):
                per_cls[names.get(i, str(i))] = float(v)
    except Exception:
        pass
    return float(r.box.map50), per_cls


# --------------------------------------------------------------------------- #
def main() -> None:
    import torch

    print("=" * 84)
    print("  ExDark 检测器微调评测（协议 A2）")
    print("  每条 pipeline 各自微调一个检测器，域内训练 + 域内测试")
    print("=" * 84)
    log(f"模式: {MODES}")
    log(f"超参: epochs={FT_EPOCHS} imgsz={FT_IMGSZ} batch={FT_BATCH}"
        f" 划分seed={FT_SEED} 训练seed={FT_SEEDS}"
        f" max_side={MAX_SIDE} limit={FT_LIMIT or '全部'}")

    # ---------------- 1. 算力 ----------------
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        dev = 0
    else:
        if not ALLOW_CPU:
            log("[!] 没有 GPU 且未设 ALLOW_CPU_TRAIN=1 —— 已停止")
            log("    到 Notebook 右侧 Settings -> Accelerator 选 GPU T4 x2")
            return
        log("[!] 使用 CPU（极慢）")
        dev = "cpu"

    # ---------------- 2. 代码与数据 ----------------
    roots = find_code_root(IN)
    if not roots:
        log("[!] 找不到代码包（需同时含 train.py 与 models/indnet.py）")
        log("    请到 Kaggle -> Datasets -> indnet-code -> New Version 上传最新包")
        return
    log(f"代码包 -> {roots[0]}")

    # 把代码复制到可写工作区（/kaggle/input 是只读的，且 sys.path 需要它）
    if FORCE or not (WORK / "models" / "indnet.py").is_file():
        if WORK.exists():
            shutil.rmtree(WORK, ignore_errors=True)
        shutil.copytree(roots[0], WORK)
        log(f"已复制代码 -> {WORK}")
    else:
        log(f"代码已就位 -> {WORK}")
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))

    img_dir, ann_csv = find_exdark()
    if img_dir is None or ann_csv is None:
        log("[!] 没找到 ExDark（图像目录 + annotations.csv）—— 请确认 Input 已挂载")
        return
    log(f"图像目录 -> {img_dir}")
    log(f"标注文件 -> {ann_csv}")

    gts, sizes = parse_annotations(ann_csv)

    from PIL import Image
    img_map = index_images(img_dir)
    stems = [s for s in sorted(gts) if s in img_map]
    log(f"图像索引 {len(img_map)} 张；可用于训练评测 {len(stems)} 张")
    if len(stems) < len(gts) * 0.9:
        log(f"  [!] 有 {len(gts) - len(stems)} 张标注找不到对应图像 —— 请核对文件名")
        if len(stems) < 50:
            log("[!] 匹配率过低，已停止")
            return

    # 标注声明的尺寸必须与实际图片一致，否则所有框错位、mAP 归零
    bad = []
    for s in stems[:200]:
        try:
            with Image.open(img_map[s]) as im:
                aw, ah = im.size
        except Exception:
            continue
        dw, dh = sizes[s]
        if abs(aw - dw) > 2 or abs(ah - dh) > 2:
            bad.append((s, (aw, ah), (dw, dh)))
    log(f"图像尺寸核对: 抽查 {min(200, len(stems))} 张，与标注不符 {len(bad)} 张")
    if bad:
        for b in bad[:5]:
            log(f"    {b[0]} 实际 {b[1]} 标注 {b[2]}")
        log("[!] 标注尺寸与实际图片不一致 —— 框会错位，已停止")
        return

    if FT_LIMIT:
        log(f"[冒烟模式] 每类只取前 {FT_LIMIT} 张")
        stems = stems[:FT_LIMIT]

    split = make_split(stems)
    log(f"划分: train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}")

    # ---------------- 3. 增强权重（只有模型模式需要）----------------
    need_model = [m for m in MODES if m in _MODEL_MODES]
    if need_model:
        restored = restore_weights_from_input()
        if restored:
            log(f"已从 Input 恢复消融权重: {restored}")
        missing = [m for m in need_model if not (ABL / m / "best.pth").is_file()]
        if missing:
            log(f"[!] 缺少增强权重 {missing}")
            log("    三个 input 里挂上上次版本的 Output 即可自动恢复（不必重训）")
            return
    ensure_ultralytics()

    # ---------------- 4. 逐 pipeline：准备数据 → 微调 → 评测 ----------------
    DS_ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    n_img = sum(len(split[k]) for k in ("train", "val", "test"))
    est_prep = (n_img * len(need_model) * 0.03 + n_img * (len(MODES) - len(need_model)) * 0.005) / 60
    n_seeds = max(1, len(FT_SEEDS))
    est_train = len(MODES) * 13 * FT_EPOCHS / 40 * n_seeds
    # 磁盘：每个 mode 一套 7362 张 JPEG（长边 <=1024），6 个 mode 约 4~6 GB
    try:
        free_gb = shutil.disk_usage("/kaggle/working").free / 1024 ** 3
        log(f"/kaggle/working 可用 {free_gb:.1f} GB")
        if free_gb < 8:
            log("  [!] 空间偏紧。可设 FT_CLEAN_TRAIN=1（每个模式训完即删训练图）"
                "或 FT_MODES 只跑子集")
    except Exception:
        pass
    log("")
    log(f"预计耗时: 图像准备 ~{est_prep:.0f} 分钟 | 微调 ~{est_train:.0f} 分钟"
        f" | 评测 ~{len(MODES) * 2 + (len(MODES) if CROSS_EVAL else 0):.0f} 分钟")

    def load_enhancer(mode: str):
        from models.indnet import INDNet

        flags = {"baseline_unet": (False, False), "plus_illum": (True, False),
                 "full_indnet": (True, True)}[mode]
        path = ABL / mode / "best.pth"
        try:
            rec = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            rec = torch.load(path, map_location="cpu")

        # 结构参数从 checkpoint 里读，避免与训练时不一致（消融时参数量本就不同）
        mcfg = ((rec.get("cfg") or {}).get("model") or {})
        state = rec.get("model") or rec.get("state_dict") or rec
        net = INDNet(base_channels=int(mcfg.get("base_channels", 32)),
                     num_blocks=int(mcfg.get("num_blocks", 4)),
                     use_illum=flags[0], use_noise=flags[1])
        net.load_state_dict({k.replace("module.", ""): v for k, v in state.items()},
                            strict=False)
        net.eval().to(dev)
        for p in net.parameters():
            p.requires_grad = False
        return net

    results, perclass, yamls, weights = {}, {}, {}, {}
    for mode in MODES:
        log("")
        log("=" * 84)
        log(f"【{mode}】{_LABEL.get(mode, mode)}")
        log("=" * 84)

        net = load_enhancer(mode) if mode in _MODEL_MODES else None
        yaml_path = prepare_dataset(mode, split, img_map, gts, net, dev)
        yamls[mode] = yaml_path
        if net is not None:
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 数据集只准备一次，用不同 seed 训练多次 —— 方差来源只有训练随机性
        results[mode], perclass[mode] = [], {}
        for seed in FT_SEEDS:
            w = finetune(mode, yaml_path, dev, seed)
            if w is None:
                log(f"    [{mode}] seed={seed} 训练失败，跳过")
                continue
            weights[(mode, seed)] = w

            m50, per_cls = evaluate(w, yaml_path, dev)
            results[mode].append(m50)
            for k, v in per_cls.items():
                perclass[mode].setdefault(k, []).append(v)
            log(f"    [{mode}] seed={seed} test mAP@0.5 = {m50:.4f}")

        if results[mode]:
            a = np.array(results[mode], dtype=float)
            sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
            log(f"    [{mode}] 汇总 {a.mean():.4f} ± {sd:.4f}"
                f"（n={len(a)}：{' / '.join(f'{v:.4f}' for v in a)}）")

        if CLEAN_TRAIN:
            for sp in ("train", "val"):
                shutil.rmtree(DS_ROOT / mode / "images" / sp, ignore_errors=True)
                shutil.rmtree(DS_ROOT / mode / "labels" / sp, ignore_errors=True)
            log("    已清理 train/val 图像（test 保留，供交叉评测）")

    # ---------------- 5. 交叉矩阵（即插即用场景）----------------
    cross = {}
    orig_seeds = [s for s in FT_SEEDS if ("original", s) in weights]
    if CROSS_EVAL and orig_seeds:
        log("")
        log("=" * 84)
        log("【交叉评测】用「原图」微调的检测器，直接测其它 pipeline 的测试图")
        log("（对应真实场景：已有检测器，只是换了个前处理）")
        log("=" * 84)
        for seed in orig_seeds:
            for mode in MODES:
                if mode == "original":
                    continue
                try:
                    m50, _ = evaluate(weights[("original", seed)], yamls[mode], dev)
                    cross.setdefault(mode, []).append(m50)
                    log(f"    [original(s{seed}) × {mode}] mAP@0.5 = {m50:.4f}")
                except Exception as exc:
                    log(f"    [{mode}] 交叉评测失败: {exc}")

    # ---------------- 6. 汇总 ----------------
    def _ms(vals):
        if not vals:
            return None, None
        a = np.array(vals, dtype=float)
        return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)

    def _w(s):
        """显示宽度：中文/全角算 2 列。"""
        return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)

    def _pad(s, width):
        return s + " " * max(0, width - _w(s))

    def _rpad(s, width):
        return " " * max(0, width - _w(s)) + s

    lines = []
    lines.append("=" * 84)
    lines.append("  表 A：域内微调协议（每条 pipeline 各自微调，公平比较）")
    if len(FT_SEEDS) > 1:
        lines.append(f"  多种子: {FT_SEEDS}（划分固定 seed={FT_SEED}，"
                     f"std 只反映训练随机性）")
    lines.append("=" * 84)
    lines.append(_pad("输入来源", 26) + _rpad("mAP@0.5", 9)
                         + _rpad("± std", 9) + _rpad("相对原图", 12))
    lines.append("-" * 84)

    base, base_sd = _ms(results.get("original") or [])
    for m in MODES:
        mean, sd = _ms(results.get(m) or [])
        if mean is None:
            lines.append(_pad(_LABEL.get(m, m), 26) + f"{'—':>9}")
            continue
        d = "" if base is None or m == "original" else f"{mean - base:>+12.4f}"
        lines.append(_pad(_LABEL.get(m, m), 26) + f"{mean:>9.4f}{sd:>9.4f}{d}")
        if len(FT_SEEDS) > 1:
            lines.append(f"    seeds: {' / '.join(f'{v:.4f}' for v in results[m])}")

    if base is not None:
        tol = max(0.005, base_sd or 0.0)
        lines.append("")
        lines.append(f"  判读阈值 = max(0.005, 种子间 std)；基线 std={base_sd:.4f}，"
                     f"故 |Δ| < {tol:.4f} 视为无差异")

    if cross:
        lines.append("")
        lines.append("=" * 84)
        lines.append("  表 B：交叉评测（检测器统一在原图训练集上微调，测试时换前处理）")
        lines.append("=" * 84)
        lines.append(_pad("输入来源", 26) + _rpad("mAP@0.5", 9)
                         + _rpad("± std", 9) + _rpad("相对原图", 12))
        lines.append("-" * 84)
        cb, cb_sd = base, (base_sd or 0.0)
        for m in MODES:
            # 原图这一行直接用表 A 的多种子结果（同一个检测器集合）
            vals = (results.get("original") or []) if m == "original" else (cross.get(m) or [])
            mean, sd = _ms(vals)
            if mean is None:
                continue
            d = "" if m == "original" else f"{mean - (cb or 0):>+12.4f}"
            lines.append(_pad(_LABEL.get(m, m), 26) + f"{mean:>9.4f}{sd:>9.4f}{d}")
            if len(vals) > 1:
                lines.append(f"    seeds: {' / '.join(f'{v:.4f}' for v in vals)}")

    if perclass:
        lines.append("")
        lines.append("=" * 84)
        lines.append("  分类型 mAP@0.5（表 A，多种子取均值）")
        lines.append("=" * 84)
        cols = [m for m in MODES if perclass.get(m)]
        lines.append(_pad("类别", 14) + "".join(f"{m:>16}" for m in cols))
        lines.append("-" * 84)
        for c in CLASS_NAMES:
            row = _pad(c, 14)
            for m in cols:
                vals = perclass[m].get(c) or []
                row += f"{np.mean(vals):>16.4f}" if vals else f"{'—':>16}"
            lines.append(row)

    lines.append("")
    lines.append("判读要点：")
    if base is not None and len(results) > 1:
        cand = [(m, _ms(results[m])[0], _ms(results[m])[1]) for m in results
                if m != "original" and results.get(m)]
        if cand:
            best, bmean, bsd = max(cand, key=lambda t: t[1])
            d = bmean - base
            tol = max(0.005, ((bsd or 0.0) + (base_sd or 0.0)) / 2)
            if d > tol:
                lines.append(f"  · 域适配后，「{_LABEL.get(best, best)}」比原图高 {d:+.4f}"
                             f"（超过阈值 {tol:.4f}）→ 增强对检测确实有帮助")
            elif d < -tol:
                lines.append(f"  · 域适配后仍全面低于原图 → 「面向检测」这条路需要重新设计")
            else:
                lines.append(f"  · 域适配后与原图基本持平（{d:+.4f}，阈值 {tol:.4f}）"
                             f"→ 增强对检测是中性，论文里不要写成「转正」")
    if cross:
        lines.append("  · 表 B 若明显低于表 A，说明「换前处理」会引入域偏移，"
                     "实际部署需要重新微调检测器 —— 这本身就是值得写进论文的发现")

    text = "\n".join(lines)
    print()
    print(text)
    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text(text + "\n", encoding="utf-8")
    log(f"结果已保存 -> {SUMMARY_TXT}")
    log("全部结束")


main()
