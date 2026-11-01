"""ExDark 暗光目标检测评测 —— 检验增强对下游检测的实际价值（论文 Table 3）。

流程：
    1. 定位 ExDark 图像目录与 annotations.csv
    2. 自适应解析 CSV（列名/坐标格式/归一化与否），**并强制预览校验**
       校验不过就停下，避免拿错误标注白跑一轮
    3. 载入 COCO 预训练 YOLOv8（**全程冻结，不参与任何训练**）
    4. 对三种输入分别推理：原图 / 增强结果（各消融变体）
    5. 计算 12 类的 mAP@0.5 并输出对比表

关键设计：
  · 检测器冻结 —— 只有这样才能证明"增益来自增强本身"，而不是检测器被重训
  · 必须包含「原图直接检测」这一行作为下限参照
  · ExDark 的 12 类与 COCO 类别一一对应，可全部评测
"""

import csv
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 必须在 import torch 之前设置：缓解显存碎片（OOM 报错里 PyTorch 官方也建议这一项）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:                      # CLAHE 需要 opencv；没有就自动跳过该条基线
    import cv2 as _cv2
except Exception:         # pragma: no cover
    _cv2 = None

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working/INDNet")
ABL = Path("/kaggle/working/ablations")
OUT = Path("/kaggle/working/detection")
FORCE_REFRESH = True
# 只评测前 N 张（0 = 全部）。第一次跑建议先设 1500 快速确认无误，再改回 0 跑全量。
LIMIT = int(os.environ.get("EXDARK_LIMIT", "0"))
# 可选：强行指定 class 列编码，取值 id0 / id1 / name（平时留空，自动判定）
CLASS_MODE_OVERRIDE = os.environ.get("EXDARK_CLASS_MODE", "").strip().lower()
# 增强时的最大边长。ExDark 混有 2560×1920 这类大图，全分辨率跑 U-Net 会打爆 15 GB 显存，
# 所以超过就先等比缩小、增强后再插值还原到原尺寸（检测器本来也会缩到 640，不影响公平性）。
ENH_MAX_SIDE = int(os.environ.get("EXDARK_ENH_MAX_SIDE", "1024"))
# 是否加经典增强基线（Gamma / CLAHE）。它们与我们自己的模型无关，
# 用来回答一个关键问题：「增强这件事本身对检测有没有好处」。
USE_CLASSICAL = os.environ.get("EXDARK_CLASSICAL", "1") == "1"
# 要跑哪些经典增强方法（逗号分隔）。默认只跑 Gamma/CLAHE 以免拖慢日常调试；
# 做「系统性对比」时显式指定全部：
#   EXDARK_CLASSICAL_MODES=gamma,clahe,histeq,ssr,msr,msrcr
CLASSICAL_WANT = [m.strip() for m in os.environ.get(
    "EXDARK_CLASSICAL_MODES", "gamma,clahe").split(",") if m.strip()]
# 检测器权重：换成 yolov8s.pt / yolov8m.pt 可以做"结论是否依赖检测器"的验证
DETECTOR_WEIGHTS = os.environ.get("EXDARK_MODEL", "yolov8n.pt")
# 图像统计用的缩略尺寸（统计只看分布，不需要原分辨率）
STAT_MAX_SIDE = int(os.environ.get("EXDARK_STAT_MAX_SIDE", "256"))
# 是否为每个学习式增强额外生成「已校色偏」的版本（{name}_align）。
# 用于检验「色偏是否是检测掉点的主因」，不需重训。
POST_ALIGN = os.environ.get("EXDARK_POST_ALIGN", "0") == "1"
# 只评测指定模式（逗号分隔），用于快速验证，例如:
#   EXDARK_MODES=original,clahe,full_indnet,full_indnet_align
MODES_FILTER = [m.strip() for m in os.environ.get("EXDARK_MODES", "").split(",") if m.strip()]

# ExDark 12 类 → COCO 类别 id（YOLO 用 0 基索引）
EXDARK_TO_COCO = {
    "bicycle": 1, "boat": 8, "bottle": 39, "bus": 5,
    "car": 2, "cat": 15, "chair": 56, "cup": 41,
    "dog": 16, "motorbike": 3, "people": 0, "table": 60,
}
# ExDark 数字编码（1-12）→ 类别名，用于 CSV 里用数字表示类别的情况
EXDARK_ID_TO_NAME = {
    1: "bicycle", 2: "boat", 3: "bottle", 4: "bus", 5: "car", 6: "cat",
    7: "chair", 8: "cup", 9: "dog", 10: "motorbike", 11: "people", 12: "table",
}
CLASS_NAMES = sorted(EXDARK_TO_COCO.keys())
NAME_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_ckpt(path):
    """兼容新旧 torch：2.6+ 把 weights_only 的默认值从 False 改成了 True。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# torch 2.x 有专门的 OutOfMemoryError；老版本只能用 RuntimeError 兜底
_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def enhance_capped(net, t: torch.Tensor, cap: int):
    """在不超过 cap 的分辨率上做增强，再插值还原到原尺寸。

    ExDark 混有 4000px 级的大图，全分辨率跑 U-Net 会直接把 15GB 显存打爆
    （实测 OOM 时报 "Tried to allocate 3.34 GiB"）。
    检测器本身也会把图缩到 640，所以这一步不影响四种输入之间的公平比较。

    返回 (增强图, 是否发生了缩放)。
    """
    h, w = t.shape[-2:]
    side = max(h, w)
    if side <= cap:
        return net.enhance(t), False
    scale = cap / side
    sh, sw = max(16, int(round(h * scale))), max(16, int(round(w * scale)))
    small = F.interpolate(t, size=(sh, sw), mode="bilinear",
                          align_corners=False, antialias=True)
    out = net.enhance(small)
    del small
    out = F.interpolate(out, size=(h, w), mode="bilinear",
                        align_corners=False, antialias=True)
    return out.clamp(0.0, 1.0), True


def _stretch_to_u8(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """float 图按通道做百分位拉伸到 uint8。

    用百分位而不是 min/max：Retinex 的输出有少量极端值，直接线性拉伸会把
    99% 的有效范围压到中间一小段，对比度全丢。
    百分位在 1/4 抽样上算，避免 4000px 大图逐像素排序拖慢整体速度。
    """
    out = np.empty(x.shape, dtype=np.float32)
    for c in range(x.shape[2]):
        ch = x[..., c]
        sample = ch[::4, ::4] if ch.shape[0] > 32 else ch
        a, b = np.percentile(sample, lo), np.percentile(sample, hi)
        if b - a < 1e-6:
            b = a + 1e-6
        out[..., c] = (ch - a) / (b - a)
    return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _retinex(rgb_f: np.ndarray, sigmas: tuple) -> np.ndarray:
    """SSR / MSR 的公共实现：log(I) − log(高斯模糊(I))，多尺度取平均。"""
    eps = 1e-4
    log_i = np.log(rgb_f + eps)
    acc = np.zeros_like(rgb_f)
    for s in sigmas:
        acc += log_i - np.log(_cv2.GaussianBlur(rgb_f, (0, 0), s) + eps)
    return acc / max(1, len(sigmas))


def _resize_long_side(rgb_u8: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """长边超过 max_side 就等比缩小，返回 (图, 缩放比)。"""
    h, w = rgb_u8.shape[:2]
    m = max(h, w)
    if _cv2 is None or max_side <= 0 or m <= max_side:
        return rgb_u8, 1.0
    s = max_side / float(m)
    small = _cv2.resize(rgb_u8, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                        interpolation=_cv2.INTER_AREA)
    return small, s


def image_stats(rgb_u8: np.ndarray, max_side: int = 256) -> dict[str, float]:
    """无参考图像统计。

    ExDark 是真实暗光拍摄、**没有正常光 GT**，所以算不了 PSNR/SSIM ——
    要解释"为什么亮化了检测反而掉点"，只能靠这些与参考图无关的量：
      Y     亮度均值（0–255）
      dark% 暗部像素占比（Y < 25.5，即 10%）
      clip% 高光饱和占比（任一通道 ≥ 250）
      grad  平均梯度幅值 —— 近似锐度/纹理保留程度
      color 色彩丰富度（Hasler–Süsstrunk 指标）
      ent   亮度直方图熵（32 bins）
    """
    x = rgb_u8
    if _cv2 is not None and max(x.shape[:2]) > max_side:
        h, w = x.shape[:2]
        s = max_side / float(max(h, w))
        x = _cv2.resize(x, (max(1, int(w * s)), max(1, int(h * s))), interpolation=_cv2.INTER_AREA)

    f = x.astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]

    gx = float(np.abs(np.diff(y, axis=1)).mean()) if y.shape[1] > 1 else 0.0
    gy = float(np.abs(np.diff(y, axis=0)).mean()) if y.shape[0] > 1 else 0.0

    rg = f[..., 0] - f[..., 1]
    yb = 0.5 * (f[..., 0] + f[..., 1]) - f[..., 2]
    colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                     + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    hist, _ = np.histogram(y, bins=32, range=(0, 256))
    p = hist.astype(np.float64) / max(1, hist.sum())
    p = p[p > 0]
    ent = float(-(p * np.log2(p)).sum())

    return {
        "Y": float(y.mean()),
        "dark%": float((y < 25.5).mean() * 100.0),
        "clip%": float((f.max(axis=2) >= 250).mean() * 100.0),
        "grad": gx + gy,
        "color": colorful,
        "ent": ent,
    }


def classical_modes(rgb_u8: np.ndarray) -> dict[str, np.ndarray]:
    """经典（非学习）增强基线，输入输出都是 HWC RGB uint8。

    这些不是可有可无的对照 —— 它们回答一个决定性的问题：
    **增强这件事本身对检测有没有好处？**
      · 若经典方法能让 mAP 高于原图 → 增强本身有益，是学习式模型的质量不够
      · 若连它们也低于原图          → 该检测器在 ExDark 上更偏好原始暗图，
        论文的叙事必须围绕"域偏移"而不是"增强质量"来写

    覆盖三类思路：逐像素映射（gamma）、局部对比度（clahe/histeq）、
    基于 Retinex 理论的照度分离（ssr / msr / msrcr）。
    """
    out = {"gamma": (np.power(rgb_u8 / 255.0, 0.5) * 255.0).round().astype(np.uint8)}
    if _cv2 is None:
        return out

    # ① 局部对比度：CLAHE 与全局直方图均衡（都在 LAB 的 L 通道上做，保住色彩）
    lab = _cv2.cvtColor(rgb_u8, _cv2.COLOR_RGB2LAB)
    clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out["clahe"] = _cv2.cvtColor(lab, _cv2.COLOR_LAB2RGB)

    lab2 = _cv2.cvtColor(rgb_u8, _cv2.COLOR_RGB2LAB)
    lab2[:, :, 0] = _cv2.equalizeHist(lab2[:, :, 0])
    out["histeq"] = _cv2.cvtColor(lab2, _cv2.COLOR_LAB2RGB)

    # ② Retinex 家族
    f = rgb_u8.astype(np.float32) / 255.0
    out["ssr"] = _stretch_to_u8(_retinex(f, (80.0,)))
    msr = _retinex(f, (15.0, 80.0, 250.0))
    out["msr"] = _stretch_to_u8(msr)
    # MSRCR：用"通道占比"做色彩恢复，抑制 MSR 常见的灰化/偏色
    s = f.sum(axis=2, keepdims=True) + 1e-4
    crf = 46.0 * (np.log(125.0 * f + 1e-4) - np.log(s))
    out["msrcr"] = _stretch_to_u8(msr * crf)
    return out


def align_color_to_ref(src_u8: np.ndarray, ref_u8: np.ndarray) -> np.ndarray:
    """把 src 的**相对色偏**校正到与 ref 一致，同时保留 src 自身的亮度。

    诊断用工具（`EXDARK_POST_ALIGN=1` 开启，会额外生成 `{name}_align` 模式）。

    ⚠️ 关于"色偏"的一个坑：绝对 R−B 会随亮度同比放大 ——
    实测我们的模型 R−B=26.8 而原图 12.5，看着像"偏色翻倍"，
    但那只是因为亮度被提亮了 2.6 倍。**按亮度归一后**
    （(R−B)/Y）各方法都在 19%~28%，并没有系统性偏色。
    所以横向比较色偏必须用归一化指标，本函数的对齐目标也是归一化色偏。

    做法：只缩放 R、B 两个通道（G 不动），令
      · (R'−B')/Y' == (R−B)/Y of ref   → 相对色偏对齐
      · Y'         == Y of src          → 提亮效果原样保留
    """
    s = src_u8.astype(np.float32)
    r = ref_u8.astype(np.float32)
    r_s = float(s[..., 0].mean()) + 1e-6
    g_s = float(s[..., 1].mean())
    b_s = float(s[..., 2].mean()) + 1e-6
    r_r = float(r[..., 0].mean())
    b_r = float(r[..., 2].mean())
    y_r = 0.299 * r_r + 0.587 * float(r[..., 1].mean()) + 0.114 * b_r
    if y_r <= 1e-6:
        return src_u8

    y_s = 0.299 * r_s + 0.587 * g_s + 0.114 * b_s
    d = y_s * (r_r - b_r) / y_r            # 目标 R'−B'（保持相对色偏不变）

    # 解 R'*? 与 B'*?:  130a − 103c = d  且  0.299·r·a + 0.114·b·c = y_s − 0.587·g
    c2 = (y_s - 0.587 * g_s - 0.299 * d) / ((0.299 + 0.114) * b_s)
    a = (d + b_s * c2) / r_s
    if not np.isfinite(a) or not np.isfinite(c2):
        return src_u8

    out = s.copy()
    out[..., 0] = np.clip(s[..., 0] * a, 0.0, 255.0)
    out[..., 2] = np.clip(s[..., 2] * c2, 0.0, 255.0)
    return out.round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# 定位
# --------------------------------------------------------------------------- #
def find_exdark() -> tuple[Path, Path] | tuple[None, None]:
    """返回 (图像目录, 标注 csv)。"""
    mount = None
    for p in IN.rglob("*"):
        if p.is_dir() and "exdark" in p.name.lower():
            mount = p if mount is None or len(str(p)) < len(str(mount)) else mount
    if mount is None:
        return None, None

    csvs = list(mount.rglob("*.csv"))
    ann = max(csvs, key=lambda f: f.stat().st_size) if csvs else None

    # 图像目录 = 含图像最多的目录
    best, best_n = None, 0
    for d in mount.rglob("*"):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if n > best_n:
            best, best_n = d, n
    return best, ann


def index_images(img_dir: Path) -> dict[str, Path]:
    """建立 {stem: path} 索引。

    直接按 stem 建索引而不是拼扩展名，因为 Kaggle 是 Linux 大小写敏感，
    数据集里混有 '.jpg' 与 '.JPG'（探查输出里就出现了 2015_00024.JPG），
    拼字符串的写法会静默漏掉这些图。
    """
    idx: dict[str, Path] = {}
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


def find_code_root(base: Path, max_depth: int = 3):
    found = []

    def scan(folder: Path, depth: int):
        try:
            # iterdir 必须放在 try 内 —— 路径不存在或中途消失时要能安全返回，
            # 否则会抛 FileNotFoundError 把整个流程打断
            if (folder / "train.py").is_file() and (folder / "models" / "indnet.py").is_file():
                found.append(folder)
                return
            if depth <= 0:
                return
            subs = [c for c in folder.iterdir() if c.is_dir()]
        except (PermissionError, OSError):
            return
        for c in subs:
            scan(c, depth - 1)

    scan(base, max_depth)
    found = sorted(set(found), key=lambda x: len(str(x)))
    # 优先用数据集里的代码，排除从「工作区 Output」挂回来的旧副本
    # （工作区目录名是大写 INDNet，数据集名为 indnet-code）
    ds = [p for p in found if "INDNet" not in str(p)]
    return ds or found


# --------------------------------------------------------------------------- #
# CSV 解析（自适应）
# --------------------------------------------------------------------------- #
# 列名别名表。匹配顺序：精确 → 前后缀，刻意不做纯子串匹配，
# 否则 'image_height' 会抢走框高 'h'、'image_width' 会抢走框宽 'w'。
IMG_ALIASES = ["image_path", "imagepath", "image_name", "image", "img_path", "img_name",
               "img", "filename", "file_name", "file", "name", "path"]
CLS_ALIASES = ["class_name", "classname", "class", "cls", "category", "label", "type",
               "object_class", "class_id", "label_id"]
XMIN_ALIASES = ["xmin", "x_min", "x_tl", "xlt", "x_topleft", "tl_x", "topleft_x",
                "x_left", "left_x", "x1", "left", "bbox_xmin",
                "x"]  # 末位的 x/y 是最后兜底：命中时按「左上角+宽高」解释并给出提示
YMIN_ALIASES = ["ymin", "y_min", "y_tl", "ytl", "y_topleft", "tl_y", "topleft_y",
                "y_top", "top_y", "y1", "top", "bbox_ymin",
                "y"]
XMAX_ALIASES = ["xmax", "x_max", "x_br", "xbr", "x_bottomright", "br_x", "bottomright_x",
                "x_right", "right_x", "x2", "right", "bbox_xmax"]
YMAX_ALIASES = ["ymax", "y_max", "y_br", "ybr", "y_bottomright", "br_y", "bottomright_y",
                "y_bottom", "bottom_y", "y2", "bottom", "bbox_ymax"]
W_ALIASES = ["w", "width", "box_w", "bbox_w", "bw"]
H_ALIASES = ["h", "height", "box_h", "bbox_h", "bh"]
IMGW_ALIASES = ["image_width", "img_width", "image_w", "img_w", "width_image", "im_w"]
IMGH_ALIASES = ["image_height", "img_height", "image_h", "img_h", "height_image", "im_h"]


def _pick(low: list[str], aliases: list[str], exclude: set) -> int | None:
    """两轮匹配：先精确相等，再前后缀。避免子串误命中。"""
    for a in aliases:
        for i, h in enumerate(low):
            if i not in exclude and h == a:
                return i
    for a in aliases:
        for i, h in enumerate(low):
            if i in exclude:
                continue
            if h.endswith("_" + a) or h.startswith(a + "_"):
                return i
    return None


def detect_columns(header: list[str]) -> dict:
    """根据列名推断各字段位置（精确/前后缀匹配，见 _pick 说明）。"""
    low = [h.strip().lower().replace(" ", "_") for h in header]

    img = _pick(low, IMG_ALIASES, set())
    if img is None:
        img = 0
    cls = _pick(low, CLS_ALIASES, {img})

    # 先锁定「图像尺寸」两列，之后所有框字段都排除它们
    img_w = _pick(low, IMGW_ALIASES, {img, cls})
    img_h = _pick(low, IMGH_ALIASES, {img, cls})
    excl = {img, cls, img_w, img_h}

    xmin = _pick(low, XMIN_ALIASES, excl)
    ymin = _pick(low, YMIN_ALIASES, excl)
    xmax = _pick(low, XMAX_ALIASES, excl)
    ymax = _pick(low, YMAX_ALIASES, excl)
    w = _pick(low, W_ALIASES, excl | {xmin, ymin, xmax, ymax})
    h = _pick(low, H_ALIASES, excl | {xmin, ymin, xmax, ymax})

    return {
        "image": img, "class": cls,
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        "w": w, "h": h, "image_w": img_w, "image_h": img_h,
    }


# 常见异名 → ExDark 12 类标准名
CLASS_ALIASES = {
    "person": "people", "pedestrian": "people", "human": "people", "humans": "people",
    "persons": "people", "peoples": "people",
    "motorcycle": "motorbike", "motor": "motorbike", "motorbikes": "motorbike",
    "bike": "bicycle", "bicycles": "bicycle", "cycle": "bicycle",
    "automobile": "car", "cars": "car",
    "tables": "table", "chairs": "chair", "cups": "cup", "bottles": "bottle",
    "boats": "boat", "buses": "bus", "cats": "cat", "dogs": "dog",
}


def _to_int(s):
    try:
        return int(float(str(s).strip()))
    except Exception:
        return None


def normalize_class(raw: str, mode: str = "name", offset: int = 0) -> str | None:
    """把 class 列的原始值转成 ExDark 标准类名。

    mode="name" → 文本类别名（含异名与模糊匹配）
    mode="id"   → 数字编码，实际类名查 EXDARK_ID_TO_NAME[v + offset]。
                  offset=0 表示 1 基（1~12），offset=1 表示 0 基（0~11）。
                  这个 offset 必须由全体数据判定，不能逐行猜——
                  曾经因为对 v=0 用 +1 兜底、对 v>=1 用原值，导致编码 0 和 1
                  同时落到 bicycle，其余整体错位一位。
    """
    s = str(raw).strip().strip('"')
    if not s:
        return None
    low = s.lower()

    if mode == "id":
        v = _to_int(s)
        if v is None:
            return None
        k = v + offset
        return EXDARK_ID_TO_NAME.get(k) or None

    # ---- 文本模式 ----
    if low in EXDARK_TO_COCO:
        return low
    if low in CLASS_ALIASES:
        return CLASS_ALIASES[low]
    for name in EXDARK_TO_COCO:
        if name in low or low in name:
            return name
    return None


def resolve_class_mode(body: list, cols: dict) -> tuple[str, int, dict]:
    """判定 class 列的编码方式，返回 (mode, offset, raw_counts)。

    判据：
      · 多数值是文本            → ("name", 0, ...)
      · 数值里出现 0 而不出现 12 → 0 基，offset=1
      · 数值里出现 12 而不出现 0 → 1 基，offset=0
    这两种情形互斥，且都能从数据本身确定，不需要人工指定。
    """
    if cols["class"] is None:
        return "unknown", 0, {}

    raw: dict = {}
    n_total = n_numeric = 0
    for r in body:
        i = cols["class"]
        if i >= len(r):
            continue
        v = r[i].strip()
        if not v:
            continue
        n_total += 1
        iv = _to_int(v)
        if iv is None:
            raw[v.lower()] = raw.get(v.lower(), 0) + 1
        else:
            n_numeric += 1
            raw[iv] = raw.get(iv, 0) + 1

    if n_total == 0:
        return "unknown", 0, raw
    if n_numeric < n_total * 0.5:
        return "name", 0, raw

    keys = {k for k in raw if isinstance(k, int)}
    if 0 in keys and 12 not in keys:
        return "id", 1, raw
    if 12 in keys and 0 not in keys:
        return "id", 0, raw
    return "unknown", 0, raw


def _looks_normalized(body: list, cols: dict, probe: int = 80) -> bool:
    """用前若干行判断坐标是否归一化（0~1）还是绝对像素。"""
    vals = []
    for r in body[:probe]:
        try:
            vals.append(float(r[cols["xmin"]]))
            vals.append(float(r[cols["ymin"]]))
            if cols["xmax"] is not None:
                vals.append(float(r[cols["xmax"]]))
                vals.append(float(r[cols["ymax"]]))
        except Exception:
            continue
    return bool(vals) and max(vals) <= 1.5


def load_annotations(csv_path: Path) -> tuple[dict, dict]:
    """返回 ({image_stem: [(class_name, x1,y1,x2,y2), ...]}, meta)。

    先打印完整预览并做三重校验，任何一项不过就停下返回空字典，
    避免拿错误解析结果白跑一轮检测。
    """
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r and any(c.strip() for c in r)]

    header, body = rows[0], rows[1:]
    cols = detect_columns(header)

    log("=" * 78)
    log("  标注解析预览（请务必核对）")
    log("=" * 78)
    log(f"  CSV: {csv_path}")
    log(f"  数据行数 {len(body)}（不含表头）")
    log(f"  表头: {header}")
    log(f"  列推断: {cols}")

    # ---- 判定框格式 ----
    if None not in (cols["xmin"], cols["ymin"], cols["xmax"], cols["ymax"]):
        box_fmt = "corner"          # (xmin, ymin, xmax, ymax)
    elif None not in (cols["xmin"], cols["ymin"], cols["w"], cols["h"]):
        box_fmt = "corner_size"     # (x, y) + (w, h)
    else:
        box_fmt = "unknown"
    log(f"  框格式: {box_fmt}")
    if box_fmt == "corner_size":
        hdr_low = [h.strip().lower() for h in header]
        if hdr_low[cols["xmin"]] in ("x", "y"):
            log("  注意: 列名为简写 x/y，按「左上角 + 宽高」解释；"
                "若你的标注实际是中心点坐标，请把输出发我。")

    log()
    log("  前 5 行原始内容:")
    for r in body[:5]:
        log(f"    {r}")

    if box_fmt == "unknown":
        log()
        log("  [!] 无法确定框格式（缺少必要列）—— 已停止，请把以上输出发给我。")
        return {}, {"box_fmt": box_fmt}

    normalized = _looks_normalized(body, cols)
    if normalized and cols["image_w"] is None:
        log()
        log("  [!] 坐标疑似归一化，但表中没有图像宽高列，无法还原为像素—— 已停止。")
        return {}, {"box_fmt": box_fmt, "normalized": True}
    log(f"  坐标尺度: {'归一化 0~1（将按图像尺寸还原）' if normalized else '绝对像素'}")

    # ---- 判定 class 编码（0 基 / 1 基 / 文本）----
    mode, offset, raw_counts = resolve_class_mode(body, cols)
    if CLASS_MODE_OVERRIDE in ("id0", "id1", "name"):
        mode, offset = ("id", 1) if CLASS_MODE_OVERRIDE == "id0" else \
                       (("id", 0) if CLASS_MODE_OVERRIDE == "id1" else ("name", 0))
        log()
        log(f"  [手动指定] class 编码 = {CLASS_MODE_OVERRIDE}（mode={mode}, offset={offset}）")
    nums = sorted((k, v) for k, v in raw_counts.items() if isinstance(k, int))
    strs = sorted((k, v) for k, v in raw_counts.items() if isinstance(k, str))
    log()
    log("  class 列编码判定:")
    if nums:
        log(f"    数字编码计数: {nums}")
    if strs:
        log(f"    文本标签计数: {strs[:20]}")
    if mode == "id":
        base = "0 基（0~11）" if offset == 1 else "1 基（1~12）"
        log(f"    判定: 数字编码，{base}；偏移 +{offset}")
        pairs = ", ".join(
            f"{k}->{EXDARK_ID_TO_NAME[k + offset]}"
            for k, _ in nums if (k + offset) in EXDARK_ID_TO_NAME
        )
        log(f"    映射: {pairs}")
    elif mode == "name":
        log("    判定: 文本类别名")
    else:
        log("    [!] 无法判定编码方式（0 与 12 同时出现或同时缺失）—— 已停止，请把输出发我。")
        return {}, {"box_fmt": box_fmt, "class_mode": mode}

    # ---- 图像尺寸分布（顺带核对 image_width/height 是否可信）----
    if cols["image_w"] is not None and cols["image_h"] is not None:
        sizes = Counter()
        for r in body[:3000]:
            try:
                sizes[(int(float(r[cols["image_w"]])), int(float(r[cols["image_h"]])))] += 1
            except Exception:
                continue
        log()
        log(f"  图像尺寸分布（前 6）: {sizes.most_common(6)}")

    # ---- 解析 ----
    gts: dict[str, list] = defaultdict(list)
    coords: list[float] = []
    deviations: list[float] = []
    worst: list[tuple[float, list]] = []
    sizes_map: dict[str, tuple[float, float]] = {}
    unknown_cls = 0
    unparsed = 0
    out_of_bounds = 0
    rescued = 0
    used = 0

    for r in body:
        try:
            stem = Path(r[cols["image"]].strip()).stem
            cls = normalize_class(r[cols["class"]], mode, offset) if cols["class"] is not None else None
            if cls is None:
                unknown_cls += 1
                continue

            x1, y1 = float(r[cols["xmin"]]), float(r[cols["ymin"]])
            if box_fmt == "corner":
                x2, y2 = float(r[cols["xmax"]]), float(r[cols["ymax"]])
            else:
                x2, y2 = x1 + float(r[cols["w"]]), y1 + float(r[cols["h"]])

            iw = ih = None
            if cols["image_w"] is not None and cols["image_h"] is not None:
                iw, ih = float(r[cols["image_w"]]), float(r[cols["image_h"]])
            if iw and ih:
                sizes_map.setdefault(stem, (iw, ih))
            if normalized and iw:
                x1, x2, y1, y2 = x1 * iw, x2 * iw, y1 * ih, y2 * ih

            # 交叉校验：角点 vs (左上角 + w/h)，两者应完全一致
            if cols["w"] is not None and cols["h"] is not None:
                try:
                    ax1, ay1 = float(r[cols["xmin"]]), float(r[cols["ymin"]])
                    aw, ah = float(r[cols["w"]]), float(r[cols["h"]])
                    if normalized and iw:
                        ax1, ay1, aw, ah = ax1 * iw, ay1 * ih, aw * iw, ah * ih
                    bx2, by2 = ax1 + aw, ay1 + ah
                    dev = max(abs(bx2 - x2), abs(by2 - y2))
                    deviations.append(dev)
                    if dev > 2.0:
                        worst.append((dev, list(r)))
                        worst.sort(key=lambda t: -t[0])
                        del worst[8:]
                    # 角点越界而 w/h 推出来的框在界内 → 用后者（并计数）
                    if iw and ih:
                        corner_ok = (x1 >= -2 and y1 >= -2 and x2 <= iw + 2 and y2 <= ih + 2)
                        size_ok = (ax1 >= -2 and ay1 >= -2 and bx2 <= iw + 2 and by2 <= ih + 2)
                        if not corner_ok and size_ok:
                            x1, y1, x2, y2 = ax1, ay1, bx2, by2
                            rescued += 1
                except Exception:
                    pass

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if x2 <= x1 or y2 <= y1:
                unparsed += 1
                continue

            # 越界框直接丢弃（列推断错误时这一项会大量触发，是重要警报）
            if iw and ih and not (x1 >= -2 and y1 >= -2 and x2 <= iw + 2 and y2 <= ih + 2):
                out_of_bounds += 1
                continue

            gts[stem].append((cls, x1, y1, x2, y2))
            coords.extend([x1, y1, x2, y2])
            used += 1
        except Exception:
            unparsed += 1
            continue

    arr = np.array(coords) if coords else np.array([0.0])
    n_bad = sum(1 for d in deviations if d > 2.0)
    meta = {
        "used": used, "unknown_class": unknown_cls, "unparsed": unparsed,
        "images": len(gts), "coord_min": float(arr.min()), "coord_max": float(arr.max()),
        "normalized": normalized, "box_fmt": box_fmt,
        "class_mode": mode, "class_offset": offset,
        "deviations": deviations, "out_of_bounds": out_of_bounds, "rescued": rescued,
        "sizes": sizes_map,
    }

    log()
    log(f"  成功解析 {used} 个框，覆盖 {len(gts)} 张图"
        f"；类别无法识别 {unknown_cls} 条；解析失败 {unparsed} 条")
    log(f"  坐标范围 [{meta['coord_min']:.2f}, {meta['coord_max']:.2f}]")
    if deviations:
        dv = np.array(deviations)
        log(f"  角点↔宽高校验: 最大 {dv.max():.2f}px | 中位 {np.median(dv):.2f}px"
            f" | 不一致(>2px) {n_bad}/{len(dv)} 条 = {n_bad / len(dv) * 100:.3f}%")
    if rescued:
        log(f"  用 w/h 修正越界角点: {rescued} 条")
    if out_of_bounds:
        log(f"  丢弃越界框: {out_of_bounds} 条")

    log()
    log("  解析结果抽样:")
    for s in sorted(gts)[:3]:
        log(f"    {s}: {len(gts[s])} 个框 -> {gts[s][:2]}")

    dist = Counter(b[0] for v in gts.values() for b in v)
    log(f"  类别分布: {dist.most_common()}")
    multi = sum(1 for v in gts.values() if len({b[0] for b in v}) > 1)
    if gts:
        log(f"  含多个不同类别的图: {multi}/{len(gts)}"
            f" = {multi / len(gts) * 100:.1f}%（官方 ExDark 每图只属一类，此项仅作参考）")

    if worst:
        log()
        log("  角点与宽高偏差最大的几行（用于定位脏数据）:")
        for dev, r in worst[:5]:
            log(f"    偏差 {dev:8.2f}px  {r}")

    # ---- 校验 ----
    n_total_rows = max(len(body), 1)
    checks = {
        "框数 > 1000":              used > 1000,
        "覆盖图数 > 1000":          len(gts) > 1000,
        "类别识别率 > 80%":         unknown_cls <= n_total_rows * 0.2,
        "坐标量级合理":             normalized or meta["coord_max"] > 2.0,
        "角点↔宽高不一致 < 1%":     n_bad <= max(10, len(deviations) * 0.01),
        "越界框 < 1%":              out_of_bounds <= max(10, n_total_rows * 0.01),
    }
    log()
    log("  校验:")
    for k, v in checks.items():
        log(f"    [{'OK ' if v else 'FAIL'}] {k}")

    if not all(checks.values()):
        log()
        log("  [!] 校验未通过，已停止 —— 请把以上全部输出发给我，据此修正解析逻辑。")
        return {}, meta

    log()
    log("  校验全部通过，继续检测评测。")
    return gts, meta


# --------------------------------------------------------------------------- #
# mAP 计算（VOC 风格全点插值）
# --------------------------------------------------------------------------- #
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0)


def compute_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map(preds: dict, gts: dict, iou_thr: float = 0.5):
    """preds: {stem: [(cls, x1,y1,x2,y2, score)]} ; gts: {stem: [(cls,x1,y1,x2,y2)]}"""
    per_class = {}
    n_pred_total = 0

    for cls in CLASS_NAMES:
        g_by_img = {k: np.array([b[1:] for b in v if b[0] == cls], dtype=np.float32) for k, v in gts.items()}
        npos = sum(len(v) for v in g_by_img.values())
        if npos == 0:
            continue

        plist = []
        for stem, boxes in preds.items():
            for b in boxes:
                if b[0] == cls:
                    plist.append((stem, b[1], b[2], b[3], b[4], b[5]))
        n_pred_total += len(plist)
        if not plist:
            per_class[cls] = 0.0
            continue

        plist.sort(key=lambda t: -t[5])
        tp = np.zeros(len(plist), dtype=np.float32)
        fp = np.zeros(len(plist), dtype=np.float32)
        matched = {k: np.zeros(len(v), dtype=bool) for k, v in g_by_img.items()}

        for i, (stem, x1, y1, x2, y2, _) in enumerate(plist):
            gt = g_by_img.get(stem, np.zeros((0, 4), dtype=np.float32))
            if len(gt) == 0:
                fp[i] = 1
                continue
            ious = iou_matrix(np.array([[x1, y1, x2, y2]], dtype=np.float32), gt)[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not matched[stem][j]:
                tp[i] = 1
                matched[stem][j] = True
            else:
                fp[i] = 1

        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        rec = tp_c / max(npos, 1)
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        per_class[cls] = compute_ap(rec, prec)

    mAP = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mAP, per_class, n_pred_total


# --------------------------------------------------------------------------- #
def main() -> None:
    os.environ.setdefault("YOLO_VERBOSE", "False")

    # ---------- 定位 ----------
    img_dir, ann_csv = find_exdark()
    if img_dir is None:
        log("[!] 没找到 ExDark 图像目录")
        return
    log(f"图像目录: {img_dir}")
    log(f"标注文件: {ann_csv}")
    if ann_csv is None:
        log("[!] 没找到 CSV 标注文件")
        return

    gts, meta = load_annotations(ann_csv)
    if not gts:
        return

    # ---------- 图像索引与尺寸核对（尽早做，失败就不必浪费后续时间）----------
    from PIL import Image

    img_map = index_images(img_dir)
    stems = [s for s in sorted(gts) if s in img_map]
    missing = len(gts) - len(stems)
    log(f"\n图像目录索引 {len(img_map)} 张；可用于评测 {len(stems)} 张"
        f"（标注中 {missing} 张找不到对应图像）")
    if not stems:
        log("[!] 标注里的图像名与目录里的文件名对不上 —— 已停止，请把上面预览发我。")
        return

    # 若标注声明的尺寸与实际图片不符，框会整体错位、mAP 直接归零，必须先拦下。
    sizes_meta = meta.get("sizes") or {}
    bad_size = []
    n_checked = 0
    for s in stems[:200]:
        if s not in sizes_meta:
            continue
        try:
            with Image.open(img_map[s]) as im:
                aw, ah = im.size
        except Exception:
            continue
        n_checked += 1
        dw_, dh_ = sizes_meta[s]
        if abs(aw - dw_) > 2 or abs(ah - dh_) > 2:
            bad_size.append((s, (aw, ah), (int(dw_), int(dh_))))
    log(f"  图像尺寸核对: 抽查 {n_checked} 张，与标注不符 {len(bad_size)} 张")
    if bad_size:
        for b in bad_size[:5]:
            log(f"    {b[0]}  实际 {b[1]}  标注 {b[2]}")
        log("  [!] 标注里的 image_width/height 与实际图片不一致 —— 边框会整体错位，已停止。")
        return

    # ---------- 准备代码与权重 ----------
    roots = find_code_root(IN)
    if not roots:
        log("[!] 找不到代码包（应同时含 train.py 与 models/indnet.py）—— 已停止")
        return
    if FORCE_REFRESH or not (WORK / "train.py").is_file():
        safe = WORK.parent
        safe.mkdir(parents=True, exist_ok=True)
        os.chdir(safe)          # 先离开待删除目录（Windows 上工作目录不能被删除）
        if WORK.exists():
            shutil.rmtree(WORK)
        shutil.copytree(roots[0], WORK)
        log(f"\n已刷新代码 -> {WORK}")
    os.chdir(WORK)
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    from models.indnet import INDNet  # noqa: E402

    # ---------- 安装 ultralytics（不动 torch）----------
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        log("\n正在安装 ultralytics ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "ultralytics"],
            check=False,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "py-cpuinfo", "psutil", "pyyaml", "requests", "scipy"],
            check=False,
        )
    import ultralytics  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    log(f"ultralytics {ultralytics.__version__} | CUDA {torch.cuda.is_available()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 冻结检测器 ----------
    detector = YOLO(DETECTOR_WEIGHTS)      # COCO 预训练
    detector.to(device)
    for p_ in detector.model.parameters():
        p_.requires_grad = False           # 冻结，不参与任何训练
    detector.model.eval()
    log(f"检测器 {DETECTOR_WEIGHTS}（COCO 预训练）已载入并冻结")

    # 只保留 ExDark 的 12 类，建立 COCO id -> ExDark 索引的映射
    coco2ex = {v: NAME_TO_IDX[k] for k, v in EXDARK_TO_COCO.items()}

    # ---------- 载入三个增强模型 ----------
    enhancers = {}
    for name, flag_i, flag_n in (
        ("baseline_unet", False, False),
        ("plus_illum", True, False),
        ("full_indnet", True, True),
    ):
        ck = ABL / name / "best.pth"
        if not ck.is_file():
            log(f"  [跳过] 缺少权重 {ck}")
            continue
        rec = load_ckpt(ck)
        m = (rec.get("cfg") or {}).get("model", {})
        net = INDNet(
            in_channels=m.get("in_channels", 3),
            base_channels=m.get("base_channels", 32),
            num_blocks=m.get("num_blocks", 4),
            use_illum=m.get("illumination_branch", flag_i),
            use_noise=m.get("noise_branch", flag_n),
        )
        net.load_state_dict(rec["model"])
        net.to(device).eval()
        enhancers[name] = net
        log(f"  已载入增强模型 {name}")

    # ---------- 外部学习式方法（Zero-DCE 等），补齐"和近年工作对比" ----------
    extra = [m.strip() for m in os.environ.get("EXDARK_EXTRA", "").split(",") if m.strip()]
    if extra:
        sys.path.insert(0, str(WORK))
        try:
            import importlib

            import external_enhancers

            importlib.reload(external_enhancers)
            for nm in extra:
                try:
                    net_x = external_enhancers.build(nm, device=device)
                    if net_x is not None:
                        enhancers[nm] = net_x
                        log(f"  已载入外部增强 {nm}")
                except Exception as exc:              # noqa: BLE001
                    log(f"  [!] 外部增强 {nm} 载入失败：{type(exc).__name__}: {exc}")
        except Exception as exc:                      # noqa: BLE001
            log(f"  [!] external_enhancers 模块不可用：{type(exc).__name__}: {exc}")

    if not enhancers:
        log()
        log("!" * 78)
        log("[!] 一个增强权重都没找到，只能评测「原图」——没有对比意义，已停止。")
        log("    期望路径: /kaggle/working/ablations/{baseline_unet,plus_illum,full_indnet}/best.pth")
        log("    常见原因: Kaggle 会话重启清空了 /kaggle/working，权重随之丢失。")
        log("    解决办法: 先跑 notebooks/kaggle_ablation_cell.py 把三个变体训出来，")
        log("              或直接跑 notebooks/kaggle_pipeline.py（自动补齐权重后再评测）。")
        log("!" * 78)
        if os.environ.get("EXDARK_ALLOW_NO_ENHANCER", "0") != "1":
            return

    # ---------- 逐图推理 ----------
    # （图像索引与尺寸核对已在前面完成）
    if LIMIT:
        stems = stems[:LIMIT]
        log(f"  [快速验证模式] 只评测前 {len(stems)} 张（改 LIMIT=0 可跑全量）")

    OUT.mkdir(parents=True, exist_ok=True)

    # 按需裁掉不需要的增强模型（快速验证时能省很多时间）
    if MODES_FILTER:
        want = set(MODES_FILTER)
        enhancers = {k: v for k, v in enhancers.items()
                     if k in want or f"{k}_align" in want}

    base_modes = ["original"]
    if USE_CLASSICAL:
        avail = {"gamma"}
        if _cv2 is not None:
            avail |= {"clahe", "histeq", "ssr", "msr", "msrcr"}
        picked = [m for m in CLASSICAL_WANT if m in avail]
        if not picked:
            picked = ["gamma"]
        base_modes += picked
    learned = list(enhancers)
    if POST_ALIGN:
        learned += [f"{n}_align" for n in enhancers]
    MODES = base_modes + learned
    if MODES_FILTER:
        MODES = [m for m in MODES if m in set(MODES_FILTER)]
    preds = {m: {} for m in MODES}
    # 无参考图像统计（ExDark 没有 GT，用这些量解释"亮化反而掉点"）
    STAT_KEYS = ["Y", "dark%", "clip%", "grad", "color", "ent"]
    stat_sum = {m: {k: 0.0 for k in STAT_KEYS} for m in MODES}
    stat_n = 0
    log(f"  评测输入共 {len(MODES)} 种: {MODES}")
    if USE_CLASSICAL:
        log(f"  经典方法统一在长边 ≤{ENH_MAX_SIDE}px 上计算后插值回原尺寸"
            f"（与学习式模型同口径；检测器本来也会缩到 640）")
    if POST_ALIGN:
        log(f"  （*_align = 保持亮度、把色偏校正到与原图一致）")

    # 显存相关统计：结果里要如实说明做过什么预处理
    n_downscaled = 0
    n_oom_retry = 0
    n_skipped = 0
    max_side_seen = 0

    for i, stem in enumerate(stems):
        path = img_map[stem]

        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        max_side_seen = max(max_side_seen, max(h, w))

        # 所有输入统一准备成 HWC RGB uint8，便于统一送检测器
        imgs: dict[str, np.ndarray] = {"original": rgb}
        if USE_CLASSICAL:
            # 与学习式模型保持同一口径：先缩到 ENH_MAX_SIDE 再增强，最后插值回原尺寸。
            # 否则 MSR/MSRCR 要在 4372px 原图上做 3 次高斯模糊 ——
            # 实测把整轮检测从约 2 小时拖到 8 小时，且对公平性没有任何好处
            # （检测器本来就会把图缩到 640）。
            small, sc = _resize_long_side(rgb, ENH_MAX_SIDE)
            for k, v in classical_modes(small).items():
                if k in preds:
                    imgs[k] = v if sc >= 1.0 else _cv2.resize(
                        v, (w, h), interpolation=_cv2.INTER_LINEAR)

        # 学习式增强放在 GPU 上做，输入按上限缩放以控制显存
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0
                             ).permute(2, 0, 1).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                for name, net in enhancers.items():
                    try:
                        out, scaled = enhance_capped(net, t, ENH_MAX_SIDE)
                    except _OOM:
                        torch.cuda.empty_cache()
                        n_oom_retry += 1
                        out, scaled = enhance_capped(net, t, max(256, ENH_MAX_SIDE // 2))
                    if scaled:
                        n_downscaled += 1
                    cur = (out[0].permute(1, 2, 0).cpu().numpy() * 255
                           ).round().astype(np.uint8)
                    imgs[name] = cur
                    if POST_ALIGN and f"{name}_align" in preds:
                        imgs[f"{name}_align"] = align_color_to_ref(cur, rgb)
                    del out
        except _OOM:
            # 降档后仍不够：跳过这张，不让整轮评测白费
            n_skipped += 1
            torch.cuda.empty_cache()
            del t
            log(f"  [!] 第 {i + 1} 张（{h}×{w}）显存不足，已跳过")
            continue
        del t

        with torch.no_grad():
            for mode in MODES:
                st = image_stats(imgs[mode], STAT_MAX_SIDE)
                for _k, _v in st.items():
                    stat_sum[mode][_k] += _v
            stat_n += 1

        with torch.no_grad():
            for mode in MODES:
                # 【重要】ultralytics 对 numpy 输入要求 HWC **BGR** 顺序
                # （官方文档: NumPy → "HWC format with BGR channels"）。
                # 之前直接传 RGB，检测器读到的是 R/B 互换的颜色，会压低绝对指标。
                img_bgr = np.ascontiguousarray(imgs[mode][..., ::-1])
                res = detector.predict(img_bgr, conf=0.01, iou=0.5, verbose=False,
                                       device=device)[0]
                boxes = []
                if res.boxes is not None and len(res.boxes):
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    conf = res.boxes.conf.cpu().numpy()
                    cl = res.boxes.cls.cpu().numpy().astype(int)
                    for (bx, cf, cc) in zip(xyxy, conf, cl):
                        if cc in coco2ex:
                            cls_name = CLASS_NAMES[coco2ex[cc]]
                            boxes.append((cls_name, float(bx[0]), float(bx[1]),
                                          float(bx[2]), float(bx[3]), float(cf)))
                preds[mode][stem] = boxes
                del img_bgr

        del imgs
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(stems)}  最大边长 {max_side_seen}px"
                f" | 缩放增强 {n_downscaled} | 降档重试 {n_oom_retry} | 跳过 {n_skipped}")
            torch.cuda.empty_cache()

    log()
    log(f"  图像最大边长 {max_side_seen}px；增强分辨率上限 {ENH_MAX_SIDE}px")
    if n_downscaled or n_oom_retry or n_skipped:
        log(f"  显存处理记录: 缩放后增强 {n_downscaled} 次 | 降档重试 {n_oom_retry} 次"
            f" | 跳过 {n_skipped} 张")
    if n_skipped:
        log("  注意：被跳过的图在 mAP 里按「全部漏检」计入，结果里已如实标注。")

    # ---------- 汇总 ----------
    log()
    log("=" * 78)
    log(f"  暗光目标检测结果（{DETECTOR_WEIGHTS} 冻结，mAP@0.5）")
    log("=" * 78)
    log(f"{'输入来源':<26} {'mAP@0.5':>10}")
    log("-" * 78)

    LABELS = {
        "original": "原图（未增强）",
        "gamma": "Gamma γ=0.5（经典）",
        "clahe": "CLAHE（经典）",
        "histeq": "直方图均衡（经典）",
        "ssr": "SSR 单尺度 Retinex",
        "msr": "MSR 多尺度 Retinex",
        "msrcr": "MSRCR（经典）",
        "zero_dce": "Zero-DCE（CVPR'20）",
        "retinexformer": "Retinexformer（ICCV'23）",
        "baseline_unet": "＋残差 U-Net 基线",
        "plus_illum": "＋光照分支",
        "full_indnet": "＋INDNet（完整）",
        "baseline_unet_align": "＋残差 U-Net（已校色偏）",
        "plus_illum_align": "＋光照分支（已校色偏）",
        "full_indnet_align": "＋INDNet｜校色偏",
    }

    summary = {}
    for mode in MODES:
        mAP, per_class, npred = evaluate_map(preds[mode], gts)
        summary[mode] = (mAP, per_class)
        label = LABELS.get(mode, mode)
        log(f"{label:<26} {mAP:>10.4f}")

    log("-" * 78)
    base = summary.get("original", (0, {}))[0]
    for mode in MODES[1:]:
        if mode in summary:
            d = summary[mode][0] - base
            log(f"  {mode:<22} 相对原图 {d:+.4f}")

    # 结论提示：经典基线决定"面向检测"这个叙事还能不能立住
    log()
    # mAP 有采样噪声：7362 张 12 类时量级约 ±0.005。
    # 小于该阈值的差异必须判为「无差异」，否则会把噪声当结论。
    TOL = 0.005
    log(f"  结论判读（差异阈值 ±{TOL}，小于它视为无差异）:")
    classic = [m for m in ("gamma", "clahe", "histeq", "ssr", "msr", "msrcr")
               if m in summary]
    if classic:
        best_c = max(summary[m][0] for m in classic)
        d_c = best_c - base
        if d_c > TOL:
            log(f"    经典增强最优 {best_c:.4f}，高于原图 {d_c:+.4f}"
                f" → 增强本身对检测有益，差距应归因于我们模型的增强质量")
        elif d_c < -TOL:
            log(f"    经典增强最优 {best_c:.4f}，低于原图 {d_c:+.4f}"
                f" → 该检测器在 ExDark 上更偏好原始暗图，"
                f"'面向检测'的叙事需要重新设计")
        else:
            log(f"    经典增强最优 {best_c:.4f}，与原图 {base:.4f} 基本持平（{d_c:+.4f}）"
                f" → 增强对这个冻结检测器几乎没有影响；"
                f"若我们的模型明显更差，问题就在增强图质量本身")
    for m in ("baseline_unet", "plus_illum", "full_indnet"):
        if m in summary:
            d = summary[m][0] - base
            verdict = "优于原图" if d > TOL else ("低于原图" if d < -TOL else "与原图持平")
            log(f"    {m:<22} {summary[m][0]:.4f}  {d:+.4f}  （{verdict}）")

    log()
    log("  分类型 mAP@0.5：")
    header = f"{'类别':<12}" + "".join(f"{m[:16]:>17}" for m in MODES)
    log(header)
    for c in CLASS_NAMES:
        row = f"{c:<12}"
        for m in MODES:
            v = summary[m][1].get(c, float("nan"))
            row += f"{v:>17.4f}" if v == v else f"{'—':>17}"
        log(row)

    # ---- 无参考图像统计：解释"亮化反而掉点" ----
    if stat_n:
        log()
        log("  图像统计（ExDark 无正常光 GT，只能用无参考量；Y 为 0–255）：")
        log(f"{'方法':<26}{'Y':>8}{'暗部%':>9}{'饱和%':>8}{'梯度':>8}{'色彩':>8}{'熵':>7}")
        log("-" * 78)
        stat_mean = {}
        for mode in MODES:
            v = {k: stat_sum[mode][k] / stat_n for k in STAT_KEYS}
            stat_mean[mode] = v
            log(f"{LABELS.get(mode, mode):<26}{v['Y']:>8.1f}{v['dark%']:>9.1f}"
                f"{v['clip%']:>8.2f}{v['grad']:>8.2f}{v['color']:>8.1f}{v['ent']:>7.2f}")
        with open(OUT / "image_stats.txt", "w", encoding="utf-8") as f:
            f.write("mode\t" + "\t".join(STAT_KEYS) + "\n")
            for mode in MODES:
                v = stat_mean[mode]
                f.write(mode + "\t" + "\t".join(f"{v[k]:.4f}" for k in STAT_KEYS) + "\n")
        log(f"  统计已保存 -> {OUT / 'image_stats.txt'}")

    Path(OUT / "detection_summary.txt").write_text(
        "\n".join(f"{k}\t{v[0]:.6f}" for k, v in summary.items()), encoding="utf-8"
    )
    log()
    log(f"结果已保存 -> {OUT / 'detection_summary.txt'}")


main()
