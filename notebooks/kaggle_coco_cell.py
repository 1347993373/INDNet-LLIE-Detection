"""方案 A：第二个检测数据集 —— COCO val2017（正常光）+ 合成暗化。

为什么这么做
------------
ExDark 上只能证明"真实暗光里增强有害"，但审稿人会问两个问题：

  ① 只有一个检测数据集？
  ② 很多论文在合成暗化/LOL 上明明报告增强有效，为什么你是负结果？

这一格用 COCO val2017 同时回答两个问题：

    original       正常光原图     → 基准（增强没道理有用，应该有害）
    original_clahe 正常光 + CLAHE → 对照：证明"有害"取决于输入本身就够亮
    dark           合成暗化       → 掉点（对照）
    dark_clahe     }              → 恢复多少比例
    dark_indnet    } 各增强方法    → 这是"为什么文献报喜、部署翻车"的关键
    dark_retinexformer }

如果合成暗化上增强能恢复大部分掉点、而真实 ExDark 上反而有害，
那就说明差异来自真实暗光的传感器噪声/运动模糊/光照分布 ——
而这正是大多数"在 LOL/合成暗化上报喜"的工作没有测的部分。

评测口径与 ExDark 完全一致
--------------------------
· 同一套 12 个类别（COCO 里全部存在，仅 people/motorbike/table 需要改名）
· 同一个冻结检测器（COCO 预训练 YOLOv8，conf 0.01 / iou 0.5）
· 同一个 mAP@0.5 实现（直接复用 kaggle_exdark_detect_cell.py 里的 evaluate_map）
· 暗化后的图仍然对全部 5000 张评测（**不筛掉没有目标的图**，否则会人为抬高 mAP）

环境变量
--------
COCO_LIMIT          只跑前 N 张（0 = 全部，冒烟时设 200）
COCO_CONDS          条件列表，逗号分隔（见上）
COCO_MODEL          检测器权重，默认 yolov8n.pt
COCO_DARK_Y         合成暗化的目标平均亮度 Y（0–255），默认 37.0（对齐 ExDark 的 37.1）
COCO_DARK_GAMMA     压暗 gamma（>1 更暗、暗部更压死），默认 1.60
                    —— 刻意不取更大值：gamma 太高会把暗部信息彻底压死，
                       实验就退化成"暗到不可恢复"的平凡结论，失去了对照意义。
COCO_DARK_NOISE     传感器噪声强度，默认 0.045（0 = 无噪声，做消融用）
COCO_ENH_MAX_SIDE   增强分辨率上限，默认 1024
COCO_FIDELITY       是否算各条件与「正常光真值」的 PSNR/SSIM，默认 1
                    —— ExDark 没有 GT 算不了；COCO 有。这一列能量化
                       "增强结果离真实正常光有多远"，再和 ΔmAP 对照
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- 参数
LIMIT = int(os.environ.get("COCO_LIMIT", "0"))
CONDS = [c.strip() for c in os.environ.get(
    "COCO_CONDS", "original,original_clahe,dark,dark_clahe,dark_indnet,dark_retinexformer"
).split(",") if c.strip()]
DETECTOR_WEIGHTS = os.environ.get("COCO_MODEL", "yolov8n.pt").strip()
DARK_Y = float(os.environ.get("COCO_DARK_Y", "37.0"))
DARK_GAMMA = float(os.environ.get("COCO_DARK_GAMMA", "1.60"))
DARK_NOISE = float(os.environ.get("COCO_DARK_NOISE", "0.045"))
CAP = int(os.environ.get("COCO_ENH_MAX_SIDE", "1024"))
FIDELITY = os.environ.get("COCO_FIDELITY", "1") == "1"
_psnr = None
_ssim = None

OUT = Path("/kaggle/working/coco")
OUT.mkdir(parents=True, exist_ok=True)

# ExDark 的 12 类，按字母序（与 evaluate_map 里的 CLASS_NAMES 一致）
CLASS_NAMES = sorted(["people", "bicycle", "car", "motorbike", "bus", "boat",
                      "bottle", "cup", "chair", "table", "cat", "dog"])
# 我们的类名 → COCO 的类名（其余同名）
OURS2COCO = {"people": "person", "motorbike": "motorcycle", "table": "dining table"}
COCO2OURS = {v: k for k, v in OURS2COCO.items()}
_OUR_SET = set(CLASS_NAMES)
# ExDark 实测参考值（image_stats 口径，Y 为 0–255）
EXDARK_REF = {"Y": 37.1, "dark%": 58.2}

_T0 = time.time()


def log(msg: str = "") -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def _w(s: str) -> int:
    """显示宽度：中文/全角算 2 列，否则表格会错位。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, width: int, align: str = "left") -> str:
    n = max(0, width - _w(s))
    return (" " * n + s) if align == "right" else (s + " " * n)


# ---------------------------------------------------------------- 依赖 / 代码根
def _ensure_ultralytics() -> None:
    """Kaggle 默认镜像没有 ultralytics。用 --no-deps 装，避免把 CUDA 版 torch 换掉。"""
    try:
        import ultralytics  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass
    log("正在安装 ultralytics（--no-deps，不动 torch）...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "ultralytics"],
                   check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "py-cpuinfo", "psutil", "pyyaml", "requests", "scipy"], check=False)


def has_code(p: Path) -> bool:
    try:
        return (p / "train.py").is_file() and (p / "models" / "indnet.py").is_file()
    except OSError:
        return False


def find_code_root() -> Path:
    cands = [Path(p).parent.parent
             for p in glob.glob("/kaggle/input/**/models/indnet.py", recursive=True)
             if has_code(Path(p).parent.parent)]
    if not cands:
        raise SystemExit("找不到代码包（train.py + models/indnet.py）；请确认 indnet-code 已挂载")
    cands.sort(key=lambda x: len(str(x)))
    return cands[0]


def load_helpers(code: Path) -> dict:
    """复用 ExDark 检测格里已经验证过的工具函数（mAP 实现必须完全一致才能横向比较）。

    那个文件是脚本、末尾无条件调用 main()，所以只取到 def main() 之前的部分来执行，
    再把需要的函数从命名空间里取出来。
    """
    hits = sorted(glob.glob("/kaggle/input/**/notebooks/kaggle_exdark_detect_cell.py",
                            recursive=True), key=len)
    if not hits:
        hits = sorted(glob.glob(str(code / "notebooks" / "kaggle_exdark_detect_cell.py")), key=len)
    if not hits:
        raise SystemExit("找不到 kaggle_exdark_detect_cell.py（mAP 实现来源）")
    src = open(hits[0], encoding="utf-8").read()
    head = src.split("def main(")[0]
    ns: dict = {}
    exec(compile(head, hits[0], "exec"), ns)  # noqa: S102
    need = ("classical_modes", "image_stats", "enhance_capped", "evaluate_map",
            "iou_matrix", "_resize_long_side")
    missing = [k for k in need if k not in ns]
    if missing:
        raise SystemExit(f"复用失败，缺少 {missing}；代码包版本不对？")
    return ns


def find_coco() -> tuple[Path, Path]:
    """返回 (instances_val2017.json, 图片目录)。"""
    anns = sorted(glob.glob("/kaggle/input/**/instances_val2017.json", recursive=True), key=len)
    if not anns:
        raise SystemExit("找不到 instances_val2017.json；请挂载 COCO val2017 数据集")
    ann = Path(anns[0])
    # 图片目录：优先与标注同级的 val2017/，其次标注文件里 file_name 的相对位置
    cands = []
    for root in (ann.parent, ann.parent.parent):
        d = root / "val2017"
        if d.is_dir():
            cands.append(d)
    cands += [Path(p) for p in glob.glob("/kaggle/input/**/val2017", recursive=True)
              if Path(p).is_dir()]
    for d in cands:
        n = len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png"))])
        if n >= 100:
            return ann, d
    raise SystemExit(f"找到标注 {ann}，但没找到 val2017 图片目录（候选: {cands[:3]}）")


# ---------------------------------------------------------------- 合成暗化
def _smooth_field(h: int, w: int, rng: np.random.Generator, grid: int = 6) -> np.ndarray:
    """低频光照场：模拟"灯只照亮一部分区域"。

    真实暗光里光照衰减是空间不均匀的（否则模型只要学一个全局增益），
    所以这里用 6×6 的随机场双线性放大，均值归一到 1。
    """
    lo = rng.uniform(0.45, 1.55, size=(grid, grid)).astype(np.float32)
    t = torch.from_numpy(lo)[None, None]
    f = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    f = np.clip(f, 0.30, 2.20)
    return f / max(float(f.mean()), 1e-6)


def _luma(x: np.ndarray) -> np.ndarray:
    """BT.601 亮度，输入 0–255 的 HWC。"""
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def synth_dark(rgb_u8: np.ndarray, seed: int,
               target_y: float | None = None) -> np.ndarray:
    """把正常光图压成"看起来像 ExDark"的暗图。

    三步物理动机：
      ① 光照场衰减（空间不均匀）
      ② gamma 压暗（暗部更压死，逼近 ExDark 的高暗部占比）
      ③ 传感器噪声（信号相关的泊松近似 + 读出噪声）—— 关键：
         真实暗光靠长曝光，噪声显著；纯 gamma 压暗不带噪声，
         这就是"合成暗化上增强有效、真实暗光上无效"的物理来源。

    最后按目标平均亮度定标，保证 Y 与 ExDark 的 37.1 可比。
    """
    ty = DARK_Y if target_y is None else target_y
    rng = np.random.default_rng(seed)
    x = rgb_u8.astype(np.float32) / 255.0
    h, w = x.shape[:2]

    x = x * _smooth_field(h, w, rng)[..., None]
    x = np.power(np.clip(x, 0.0, 1.0), DARK_GAMMA)

    # 第一次定标（噪声会略抬高亮度，稍后再校一次）
    cur = float(_luma(x * 255.0).mean())
    if cur > 1e-6:
        x = np.clip(x * (ty / cur), 0.0, 1.0)

    if DARK_NOISE > 0:
        n = rng.normal(0.0, 1.0, x.shape).astype(np.float32)
        x = x + n * (DARK_NOISE * np.sqrt(np.clip(x, 0.0, 1.0)) + 0.006)
        x = np.clip(x, 0.0, 1.0)

    cur = float(_luma(x * 255.0).mean())
    if cur > 1e-6:
        x = np.clip(x * (ty / cur), 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------- COCO 标注
def load_coco(ann_path: Path, img_dir: Path) -> tuple[list[str], dict, dict]:
    """返回 (全部 stem, {stem: [(cls, x1,y1,x2,y2)]} 正常 GT, {stem: [...]} crowd 区)。

    类别 id 一律从 JSON 的 categories 读（Kaggle 上的 COCO 子集经常重编号，
    写死 1/2/3 会直接全军覆没）。

    iscrowd 单独返回、不进 GT：crowd 是一个"一群人"的大框，当成普通目标会让
    完美检测也拿不到 mAP=1（实测 0.77）。COCO 官方口径是把它当忽略区 ——
    落进 crowd 区的检测既不算 TP 也不算 FP，这里在预测阶段按 IoU>0.5 丢弃。
    """
    ann = json.load(open(ann_path, encoding="utf-8"))
    name_by_cid = {c["id"]: c["name"] for c in ann["categories"]}
    cid2our = {}
    for cid, nm in name_by_cid.items():
        o = COCO2OURS.get(nm, nm)
        if o in _OUR_SET:
            cid2our[cid] = o

    if not cid2our:
        raise SystemExit(f"标注里没有任何一个目标类别；categories 前 8 个: "
                         f"{list(name_by_cid.items())[:8]}")

    stem_by_iid = {im["id"]: Path(im["file_name"]).stem for im in ann["images"]}
    on_disk = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    stems, gts, crowd = [], {}, {}
    for im in ann["images"]:
        s = Path(im["file_name"]).stem
        if s in on_disk:
            stems.append(s)
    stems.sort()

    n_crowd = 0
    for a in ann["annotations"]:
        o = cid2our.get(a["category_id"])
        if o is None:
            continue
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        s = stem_by_iid.get(a["image_id"])
        if s is None:
            continue
        box = (o, float(x), float(y), float(x + w), float(y + h))
        if a.get("iscrowd", 0):
            crowd.setdefault(s, []).append(box)
            n_crowd += 1
        else:
            gts.setdefault(s, []).append(box)

    log(f"标注 {ann_path.name}：{len(stems)} 张图可用 / 标注 {len(ann['annotations'])} 条")
    log(f"类别映射 {len(cid2our)} 类：{sorted(cid2our.values())}")
    n_box = sum(len(v) for v in gts.values())
    log(f"用于评测的框 {n_box} 个，覆盖 {len(gts)} 张图；"
        f"其余 {len(stems) - len(gts)} 张无目标（保留，用于统计误检）")
    log(f"crowd 区 {n_crowd} 个（落在其中的检测不计 TP/FP，与 COCO 口径一致）")
    return stems, gts, crowd


def _drop_in_crowd(boxes: list, crowd_boxes: list, iou_fn) -> list:
    """把落在 crowd 区（同类、IoU>0.5）的检测丢掉，模拟 COCO 的 ignore 语义。"""
    if not crowd_boxes or not boxes:
        return boxes
    keep = []
    for b in boxes:
        drop = False
        for c in crowd_boxes:
            if c[0] != b[0]:
                continue
            ious = iou_fn(np.array([[b[1], b[2], b[3], b[4]]], dtype=np.float32),
                          np.array([[c[1], c[2], c[3], c[4]]], dtype=np.float32))
            if float(ious[0, 0]) > 0.5:
                drop = True
                break
        if not drop:
            keep.append(b)
    return keep


# ---------------------------------------------------------------- 增强调度
def _make_enhancers(conds: list[str], code: Path, device: str, helpers: dict) -> dict:
    """按需载入：用到什么载什么，省显存也省调试时间。"""
    from models.indnet import INDNet  # noqa: PLC0415

    enh: dict = {}

    if any(c.endswith("_indnet") for c in conds):
        cks = sorted(glob.glob("/kaggle/working/**/full_indnet/best.pth", recursive=True)
                     + glob.glob("/kaggle/input/**/full_indnet/best.pth", recursive=True),
                     key=len)
        if not cks:
            raise SystemExit("找不到 full_indnet/best.pth；请把带权重的 notebook 版本挂到 Input")
        rec = torch.load(cks[0], map_location="cpu", weights_only=False)
        mc = (rec.get("cfg") or {}).get("model", {})
        net = INDNet(in_channels=mc.get("in_channels", 3),
                     base_channels=mc.get("base_channels", 48),
                     num_blocks=mc.get("num_blocks", 6),
                     use_illum=mc.get("illumination_branch", True),
                     use_noise=mc.get("noise_branch", True))
        net.load_state_dict(rec["model"])
        net.to(device).eval()
        enh["indnet"] = net
        log(f"INDNet 已载入（{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M）<- {cks[0]}")

    for nm in ("retinexformer", "zero_dce"):
        if not any(c.endswith("_" + nm) for c in conds):
            continue
        try:
            sys.path.insert(0, str(code / "notebooks"))
            import external_enhancers as ee  # noqa: PLC0415
            net = ee.build(nm, device=device)
            if net is not None:
                enh[nm] = net
                log(f"{nm} 已载入（{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M）")
        except Exception as exc:  # noqa: BLE001
            log(f"  [!] {nm} 载入失败：{type(exc).__name__}: {exc}（该条件下会留空）")
    return enh


def _apply(method: str, rgb: np.ndarray, enh: dict, device: str,
           helpers: dict) -> np.ndarray | None:
    """对一张 uint8 HWC 图施加某种增强，返回 uint8 HWC。"""
    if method == "none":
        return rgb
    if method in ("gamma", "clahe", "histeq", "ssr", "msr", "msrcr"):
        small, sc = helpers["_resize_long_side"](rgb, CAP)
        out = helpers["classical_modes"](small).get(method)
        if out is None:
            return None
        if sc < 1.0:
            import cv2
            out = cv2.resize(out, (rgb.shape[1], rgb.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        return out
    net = enh.get(method)
    if net is None:
        return None
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    try:
        with torch.no_grad():
            out, _ = helpers["enhance_capped"](net, t, CAP)
    except Exception as exc:  # noqa: BLE001
        torch.cuda.empty_cache()
        log(f"  [!] {method} 前向失败：{type(exc).__name__}: {exc}")
        return None
    cur = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    del t, out
    return cur


def _parse_cond(cond: str) -> tuple[str, str]:
    """'original' / 'dark_clahe' → ('original'|'dark', method)"""
    if cond in ("original", "dark"):
        return cond, "none"
    base, _, method = cond.partition("_")
    if base not in ("original", "dark"):
        raise SystemExit(f"条件名无法解析: {cond}（应以 original_ / dark_ 开头）")
    return base, method


# ---------------------------------------------------------------- 主流程
def main() -> None:
    os.environ.setdefault("YOLO_VERBOSE", "False")
    _ensure_ultralytics()

    code = find_code_root()
    sys.path.insert(0, str(code))
    log(f"代码根 {code}")

    helpers = load_helpers(code)
    # mAP 必须按同一套 12 类算；这里显式覆写，避免依赖被复用模块的默认值
    helpers["CLASS_NAMES"] = CLASS_NAMES
    log("已复用 ExDark 检测格的 mAP / 图像统计 / 经典增强实现")

    # 保真度（与正常光真值比）—— 仅 COCO 有 GT，这一点 ExDark 做不到
    global _psnr, _ssim
    if FIDELITY:
        try:
            from utils.metrics import calculate_psnr as _p  # noqa: PLC0415
            from utils.metrics import calculate_ssim as _s  # noqa: PLC0415
            _psnr, _ssim = _p, _s
            log("保真度指标：PSNR + SSIM（对照正常光原图）")
        except Exception as exc:  # noqa: BLE001
            _psnr = _ssim = None
            log(f"  [!] 保真度指标不可用（{type(exc).__name__}: {exc}），跳过该列")

    ann_path, img_dir = find_coco()
    log(f"标注 {ann_path}")
    log(f"图片目录 {img_dir}")
    stems, gts, crowd = load_coco(ann_path, img_dir)

    if LIMIT > 0:
        stems = stems[:LIMIT]
        log(f"COCO_LIMIT={LIMIT} → 本次只跑前 {LIMIT} 张")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"设备 {device}")

    from ultralytics import YOLO  # noqa: PLC0415

    det = YOLO(DETECTOR_WEIGHTS)
    det.to(device)
    for p_ in det.model.parameters():
        p_.requires_grad = False
    det.model.eval()
    log(f"检测器 {DETECTOR_WEIGHTS} 已载入并冻结")

    # 检测器输出的是 80 类索引，按名字映射回我们的 12 类
    idx2our = {}
    for i, nm in (det.names or {}).items():
        o = COCO2OURS.get(nm, nm)
        if o in _OUR_SET:
            idx2our[int(i)] = o
    if not idx2our:
        raise SystemExit(f"检测器类别名与目标 12 类对不上；names 前 8: {list((det.names or {}).items())[:8]}")
    log(f"检测器类别映射 {len(idx2our)} 类: {sorted(idx2our.values())}")

    enh = _make_enhancers(CONDS, code, device, helpers)

    # 预检：每个条件是否可产出
    ok_conds = []
    for c in CONDS:
        _, m = _parse_cond(c)
        if m == "none" or m in enh or m in ("gamma", "clahe", "histeq", "ssr", "msr", "msrcr"):
            ok_conds.append(c)
        else:
            log(f"  [跳过] 条件 {c}：方法 {m} 不可用")
    if "original" not in ok_conds:
        raise SystemExit("条件列表里必须有 original（基准）")
    log(f"评测输入共 {len(ok_conds)} 种: {ok_conds}")

    STAT_KEYS = ["Y", "dark%", "clip%", "grad", "color", "ent"]
    preds = {c: {} for c in ok_conds}
    stat_sum = {c: {k: 0.0 for k in STAT_KEYS} for c in ok_conds}
    fid_sum = {c: {"psnr": 0.0, "ssim": 0.0} for c in ok_conds}
    n_fid = 0
    n_stat = 0
    n_missing = 0
    samples: list[tuple[str, dict]] = []      # 画 sanity 图用

    log("")
    log("=" * 78)
    t_start = time.time()
    for i, stem in enumerate(stems, 1):
        path = None
        for ext in (".jpg", ".jpeg", ".png"):
            q = img_dir / f"{stem}{ext}"
            if q.is_file():
                path = q
                break
        if path is None:
            n_missing += 1
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

        seed = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16)
        dark = None
        imgs: dict[str, np.ndarray] = {}
        for c in ok_conds:
            base, method = _parse_cond(c)
            if base == "dark" and dark is None:
                dark = synth_dark(rgb, seed)
            src = rgb if base == "original" else dark
            out = _apply(method, src, enh, device, helpers)
            if out is None:
                continue
            imgs[c] = out

            if len(samples) < 3 and base == "dark" and method == "none":
                samples.append((stem, {"original": rgb, "dark": out}))

        for c, v in imgs.items():
            st = helpers["image_stats"](v, 256)
            for k in STAT_KEYS:
                stat_sum[c][k] += st[k]
            preds[c][stem] = []
        n_stat += 1

        # 与「正常光真值」的保真度 —— 只有 COCO 能做（ExDark 没有 GT）
        if FIDELITY and _psnr is not None:
            ref = rgb.astype(np.float32) / 255.0
            for c, v in imgs.items():
                if c == "original" or v.shape != rgb.shape:
                    continue
                cur = v.astype(np.float32) / 255.0
                fid_sum[c]["psnr"] += _psnr(cur, ref)
                if _ssim is not None:
                    try:
                        fid_sum[c]["ssim"] += _ssim(cur, ref)
                    except Exception:  # noqa: BLE001  （skimage 缺失等）
                        _ssim = None
                        log("  [!] SSIM 不可用，后续只统计 PSNR")
            n_fid += 1

        for c, v in imgs.items():
            img_bgr = np.ascontiguousarray(v[..., ::-1])   # ultralytics 要 BGR
            res = det.predict(img_bgr, conf=0.01, iou=0.5, verbose=False, device=device)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf_ = res.boxes.conf.cpu().numpy()
                cl = res.boxes.cls.cpu().numpy().astype(int)
                for bx, cf, cc in zip(xyxy, conf_, cl):
                    o = idx2our.get(int(cc))
                    if o is None:
                        continue
                    boxes.append((o, float(bx[0]), float(bx[1]),
                                  float(bx[2]), float(bx[3]), float(cf)))
            if boxes:
                boxes = _drop_in_crowd(boxes, crowd.get(stem, []), helpers["iou_matrix"])
            preds[c][stem] = boxes

        del imgs, rgb, dark
        if i % 200 == 0:
            el = time.time() - t_start
            eta = el / i * (len(stems) - i)
            log(f"  {i}/{len(stems)}  已用 {el / 60:.1f} min  预计剩余 {eta / 60:.0f} min")
            torch.cuda.empty_cache()

    log(f"  完成 {len(stems)} 张，耗时 {(time.time() - t_start) / 60:.1f} min")
    if n_missing:
        log(f"  [!] {n_missing} 张在图片目录里找不到，已跳过")

    # ---------- 汇总 ----------
    summary = {}
    for c in ok_conds:
        mAP, per_class, npred = helpers["evaluate_map"](preds[c], gts)
        summary[c] = (mAP, per_class, npred)

    stats = {c: {k: stat_sum[c][k] / max(n_stat, 1) for k in STAT_KEYS} for c in ok_conds}

    LINES = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        LINES.append(s)

    emit()
    emit("=" * 80)
    emit(f"  COCO val2017 · 正常光 + 合成暗化（{DETECTOR_WEIGHTS} 冻结，mAP@0.5）")
    emit("=" * 80)
    emit(_pad("条件", 30) + _pad("mAP@0.5", 10, "right")
         + _pad("Δ vs 正常光", 14, "right") + _pad("Δ vs 暗化", 12, "right"))
    emit("-" * 80)
    LAB = {
        "original": "正常光原图",
        "original_clahe": "正常光 + CLAHE",
        "original_indnet": "正常光 + INDNet",
        "dark": "合成暗化",
        "dark_clahe": "暗化 + CLAHE",
        "dark_indnet": "暗化 + INDNet",
        "dark_retinexformer": "暗化 + Retinexformer",
        "dark_zero_dce": "暗化 + Zero-DCE",
        "dark_gamma": "暗化 + Gamma",
        "dark_msr": "暗化 + MSR",
        "dark_histeq": "暗化 + 直方图均衡",
    }
    base_map = summary["original"][0]
    dark_map = summary["dark"][0] if "dark" in summary else None
    for c in ok_conds:
        m = summary[c][0]
        d1 = f"{m - base_map:+.4f}"
        d2 = f"{m - dark_map:+.4f}" if dark_map is not None else "—"
        emit(_pad(LAB.get(c, c), 30) + _pad(f"{m:.4f}", 10, "right")
             + _pad(d1, 14, "right") + _pad(d2, 12, "right"))

    emit()
    if dark_map is not None and base_map - dark_map > 1e-6:
        emit("  合成暗化造成的掉点，各方法恢复比例（(mAP−暗化)/(正常光−暗化)）:")
        for c in ok_conds:
            if not c.startswith("dark_") or c == "dark":
                continue
            rec = (summary[c][0] - dark_map) / (base_map - dark_map)
            emit(f"    {_pad(LAB.get(c, c), 26)}{rec * 100:6.1f}%")
    emit(f"\n  判读阈值：COCO 样本量 ~{len(stems)} 张，mAP 采样噪声量级约 "
         f"±{0.005 * (7362 / max(len(stems), 1)) ** 0.5:.4f}；小于它视为无差异")

    # 自动判读（数字全部来自本次实测，避免手写结论与数据不符）
    emit()
    emit("  判读：")
    oc = [c for c in ok_conds if c.startswith("original_")]
    if oc:
        worst = min(oc, key=lambda c: summary[c][0])
        dv = summary[worst][0] - base_map
        verdict = ("为负 ⇒ 输入已经够亮时增强无益，与双因子模型一致" if dv < -0.005
                   else ("为正 ⇒ 增强在正常光上也有收益，需在 §7 讨论" if dv > 0.005
                         else "基本无差异（小于判读阈值）"))
        emit(f"    · 正常光图上施加 {len(oc)} 种增强，最低的是 {LAB.get(worst, worst)}："
             f"{dv:+.4f} —— {verdict}")
    if dark_map is not None and base_map - dark_map > 1e-6:
        rcv = {c: (summary[c][0] - dark_map) / (base_map - dark_map)
               for c in ok_conds if c.startswith("dark_") and c != "dark"}
        if rcv:
            best = max(rcv, key=rcv.get)
            emit(f"    · 合成暗化后掉 {base_map - dark_map:.4f}；"
                 f"恢复最好的是 {LAB.get(best, best)}，恢复了 {rcv[best] * 100:.1f}%")
            if rcv[best] > 0.5:
                emit("    · 恢复过半 ⇒ 合成暗化损失的信息是可恢复的；"
                     "与真实暗光（ExDark）上的负结果形成对照")
            else:
                emit("    · 恢复不足一半 ⇒ 即使原始信息本可恢复，增强也没能救回来；")
                emit("      说明瓶颈不只在\"真实噪声\"，还在增强本身的复原精度")
                emit("      （对照：正常光原图 = 完美复原的上界，缺口仍有 "
                     f"{base_map - summary[best][0]:.4f}）")
        else:
            emit("    · 暗化后没有任何增强条件，无法计算恢复比例")

    # 图像统计
    emit()
    emit("  图像统计（Y 为 0–255；暗部% 与 ExDark 的 58.2 对照）")
    emit(_pad("条件", 26) + _pad("Y", 8, "right") + _pad("暗部%", 9, "right")
         + _pad("饱和%", 9, "right") + _pad("梯度", 8, "right")
         + _pad("色彩", 8, "right") + _pad("熵", 7, "right"))
    emit("-" * 80)
    for c in ok_conds:
        s = stats[c]
        emit(_pad(LAB.get(c, c), 26) + _pad(f"{s['Y']:.1f}", 8, "right")
             + _pad(f"{s['dark%']:.1f}", 9, "right") + _pad(f"{s['clip%']:.2f}", 9, "right")
             + _pad(f"{s['grad']:.2f}", 8, "right") + _pad(f"{s['color']:.1f}", 8, "right")
             + _pad(f"{s['ent']:.2f}", 7, "right"))
    emit(_pad("（ExDark 实测参照）", 26) + _pad(f"{EXDARK_REF['Y']:.1f}", 8, "right")
         + _pad(f"{EXDARK_REF['dark%']:.1f}", 9, "right"))

    # 保真度（有 GT 才算得出）
    if n_fid and _psnr is not None:
        emit()
        emit("  与「正常光真值」的保真度（ExDark 没有 GT，这一列只有 COCO 能给）")
        emit(_pad("条件", 26) + _pad("PSNR↑", 9, "right") + _pad("SSIM↑", 9, "right")
             + _pad("mAP", 9, "right") + _pad("ΔmAP vs 暗化", 14, "right"))
        emit("-" * 78)
        pts, fid = [], {}
        for c in ok_conds:
            if c == "original":
                continue
            ps = fid_sum[c]["psnr"] / n_fid
            sm = fid_sum[c]["ssim"] / n_fid
            d = summary[c][0] - dark_map if dark_map is not None else float("nan")
            fid[c] = (ps, sm)
            emit(_pad(LAB.get(c, c), 26) + _pad(f"{ps:.2f}", 9, "right")
                 + _pad(f"{sm:.4f}" if _ssim is not None else "—", 9, "right")
                 + _pad(f"{summary[c][0]:.4f}", 9, "right") + _pad(f"{d:+.4f}", 14, "right"))
            pts.append((ps, summary[c][0]))

        def _corr(xs, ys):
            if len(xs) < 3:
                return float("nan")
            return float(np.corrcoef(np.array(xs, float), np.array(ys, float))[0, 1])

        def _fmt_r(v):
            return "—" if v != v else f"{v:+.3f}"

        # 全体相关会被「正常光 + 增强」那一点拉高（它既近真值又高分），
        # 判读必须用暗化分支内部的相关 —— 否则会得出相反结论
        dk = [c for c in fid if c.startswith("dark")]
        if len(pts) >= 3:
            emit(f"\n  PSNR vs mAP：全体 r={_fmt_r(_corr([p for p, _ in pts], [m for _, m in pts]))}"
                 f"（n={len(pts)}）｜仅暗化分支 r="
                 f"{_fmt_r(_corr([fid[c][0] for c in dk], [summary[c][0] for c in dk]))}（n={len(dk)}）")
            emit("  ⚠ 全体 r 被「正常光+增强」那点拉高，判读请只看暗化分支")
            if _ssim is not None:
                emit(f"  SSIM vs mAP：仅暗化分支 r="
                     f"{_fmt_r(_corr([fid[c][1] for c in dk], [summary[c][0] for c in dk]))}")

        # 直接找矛盾：保真度升高但检测变差的案例（比相关系数有说服力得多）
        if "dark" in fid:
            p0, s0 = fid["dark"]
            m0 = summary["dark"][0]
            for c in dk:
                if c == "dark":
                    continue
                dp, ds = fid[c][0] - p0, (fid[c][1] - s0 if _ssim is not None else 0.0)
                dm = summary[c][0] - m0
                if dp > 0.3 and dm < 0:
                    emit(f"  · 反例 {LAB.get(c, c)}：PSNR {dp:+.2f} dB、SSIM {ds:+.3f}，"
                         f"但 mAP {dm:+.4f} ⇒ PSNR/SSIM 不能预测检测")

    # 分类型
    emit()
    emit("  分类型 mAP@0.5（列 = 条件）")
    emit(_pad("类别", 12) + "".join(_pad(LAB.get(c, c), 18, "right") for c in ok_conds))
    emit("-" * (12 + 18 * len(ok_conds)))
    for cls in CLASS_NAMES:
        row = _pad(cls, 12)
        for c in ok_conds:
            pc = summary[c][1]
            row += _pad(f"{pc[cls]:.4f}", 18, "right") if cls in pc else _pad("—", 18, "right")
        emit(row)

    # sanity 图：暗化是否看起来合理
    try:
        if samples:
            cols = ["original", "dark"]
            pick = next((c for c in ok_conds if c.startswith("dark_") and c != "dark"), None)
            W, H = 360, 270
            canvas = Image.new("RGB", (W * len(cols), H * len(samples)), "white")
            dr = ImageDraw.Draw(canvas)
            for r, (stem, pair) in enumerate(samples):
                for cc, name in enumerate(cols):
                    im = Image.fromarray(pair[name]).resize((W, H), Image.LANCZOS)
                    canvas.paste(im, (cc * W, r * H))
            p = OUT / "coco_dark_sanity.png"
            canvas.save(p)
            log(f"暗化效果抽样图 -> {p}（左=正常光，右=合成暗化；列宽 {W}，共 {len(samples)} 行）")
            if pick:
                log(f"  （想看增强后的效果，把条件列表里的 {pick} 放到前面再跑 200 张即可）")
    except Exception as exc:  # noqa: BLE001
        log(f"[!] sanity 图生成失败：{type(exc).__name__}: {exc}")

    (OUT / "coco_summary.txt").write_text("\n".join(LINES), encoding="utf-8")
    log(f"结果已保存 -> {OUT / 'coco_summary.txt'}")


if __name__ == "__main__":
    main()
