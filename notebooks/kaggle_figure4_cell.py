"""Fig. 4：定性对比图 —— 原图 / CLAHE / Retinexformer / INDNet 的检测结果。

跑法：Kaggle Notebook，Accelerator 选 **None (CPU)**，Internet 开（Retinexformer 要下权重）。
约 2–3 分钟，不占 GPU 额度。

产物：/kaggle/working/figures/fig4_qualitative.png
      （4 行 × 4 列：每行一张图，每列一种输入，框为该输入下的 YOLOv8n 检测结果）

选图逻辑不是随机：先在小样本上统计「原图检测数 − INDNet 检测数」，
挑掉得最多的 4 张 —— 这样图里能直接看到"增强后漏检"的现象，而不是挑好看的。
"""

import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

CONF = 0.25          # 画图用的置信度阈值（比评测的 0.01 高，图才干净）
MAX_SIDE = 1024
PANEL_W = 420
N_SAMPLE = 30        # 用来挑图的候选数
N_PICK = 4           # 最终画几张

OUT = Path("/kaggle/working/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 环境
def has_code(p: Path) -> bool:
    try:
        return (p / "train.py").is_file() and (p / "models" / "indnet.py").is_file()
    except OSError:
        return False


cands = [Path(p).parent.parent
         for p in glob.glob("/kaggle/input/**/models/indnet.py", recursive=True)
         if has_code(Path(p).parent.parent)]
if not cands:
    raise SystemExit("找不到代码包（train.py + models/indnet.py），请确认 indnet-code 已挂载")
cands.sort(key=lambda x: len(str(x)))
CODE = cands[0]
sys.path.insert(0, str(CODE))
print("代码根:", CODE)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("设备:", dev)

# ---------------------------------------------------------------- 模型
from models.indnet import INDNet  # noqa: E402

cks = sorted(glob.glob("/kaggle/working/**/full_indnet/best.pth", recursive=True)
             + glob.glob("/kaggle/input/**/full_indnet/best.pth", recursive=True), key=len)
if not cks:
    raise SystemExit("找不到 full_indnet/best.pth，请把带权重的版本挂到 Input")
print("权重:", cks[0])

rec = torch.load(cks[0], map_location="cpu", weights_only=False)
mc = rec["cfg"]["model"]
ours = INDNet(in_channels=3,
              base_channels=mc.get("base_channels", 48),
              num_blocks=mc.get("num_blocks", 6),
              use_illum=mc.get("illumination_branch", True),
              use_noise=mc.get("noise_branch", True))
ours.load_state_dict(rec["model"])
ours.to(dev).eval()
print(f"INDNet 已载入（{sum(p.numel() for p in ours.parameters()) / 1e6:.2f} M）")

# Retinexformer（可选：拿不到就只用 CLAHE + 我们）
rf = None
try:
    sys.path.insert(0, str(CODE / "notebooks"))
    import external_enhancers as ee
    rf = ee.build("retinexformer", device=dev)
except Exception as exc:  # noqa: BLE001
    print(f"[!] Retinexformer 不可用（{type(exc).__name__}: {exc}），该列将留空")

# ultralytics 在 Kaggle 默认镜像里没有，必须自己装（与检测格用同一套参数）
try:
    import ultralytics  # noqa: F401
except ImportError:
    import subprocess
    print("正在安装 ultralytics ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "ultralytics"],
                   check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "py-cpuinfo", "psutil", "pyyaml", "requests", "scipy"], check=False)

from ultralytics import YOLO  # noqa: E402

det = YOLO("yolov8n.pt")
det.to(dev)

# ---------------------------------------------------------------- 工具
def enhance(net, rgb: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
    h, w = t.shape[-2:]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1.0:
        t = F.interpolate(t, size=(max(16, int(h * s)), max(16, int(w * s))),
                          mode="bilinear", align_corners=False, antialias=True)
    with torch.no_grad():
        o = net.enhance(t)
    if s < 1.0:
        o = F.interpolate(o, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    return (o[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def detect(rgb: np.ndarray):
    """返回 (带框的 RGB 图, 框数)。ultralytics 的 numpy 输入要 HWC BGR。"""
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    r = det.predict(bgr, conf=CONF, iou=0.5, verbose=False, device=dev)[0]
    vis = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB) if len(r.boxes) else rgb
    return vis, int(len(r.boxes))


# ---------------------------------------------------------------- 找数据
# ⚠️ 注意：这里必须用 len(str(x))，不能用 key=len —— Path 对象没有 __len__，
# 会抛 "object of type 'PosixPath' has no len()"。
hits = sorted({Path(p).parent.parent
               for p in glob.glob("/kaggle/input/**/Dataset/Dataset", recursive=True)},
              key=lambda x: len(str(x)))
if not hits:
    hits = sorted({Path(p).parent
                   for p in glob.glob("/kaggle/input/**/exdark*/**/*.jpg", recursive=True)},
                  key=lambda x: len(str(x)))
IMG_DIR = hits[0] if hits else None
if IMG_DIR is None:
    raise SystemExit("找不到 ExDark 图像目录")

files = sorted([p for p in Path(IMG_DIR).rglob("*") if p.suffix.lower() in (".jpg", ".png")])
print(f"ExDark 图像 {len(files)} 张，目录 {IMG_DIR}")

step = max(1, len(files) // N_SAMPLE)
sample = files[::step][:N_SAMPLE]

# ---------------------------------------------------------------- 挑掉点最明显的 4 张
scored = []
for f in sample:
    rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
    _, n0 = detect(rgb)
    _, n1 = detect(enhance(ours, rgb))
    scored.append((n0 - n1, n0, n1, f))

scored.sort(key=lambda x: -x[0])
print("\n候选（原图框数 → 我们增强后框数，差越大越能说明问题）：")
for d, n0, n1, f in scored[:N_PICK]:
    print(f"  {f.name:20s} {n0:2d} -> {n1:2d}  (差 {d:+d})")

picks = [f for d, n0, n1, f in scored[:N_PICK]]

# ---------------------------------------------------------------- 排版
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except OSError:
    FONT = ImageFont.load_default()

COLS = [("Original", "orig"), ("CLAHE", "clahe"), ("Retinexformer", "rf"), ("INDNet (Ours)", "ours")]

rows = []
for f in picks:
    rgb = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
    imgs = {"orig": rgb, "clahe": clahe(rgb),
            "ours": enhance(ours, rgb),
            "rf": enhance(rf, rgb) if rf is not None else None}
    panels = []
    for label, key in COLS:
        if imgs[key] is None:
            panels.append((label, None, 0))
            continue
        vis, n = detect(imgs[key])
        panels.append((label, vis, n))
    rows.append((f.stem, panels))
    print(f"  已处理 {f.stem}")

h0, w0 = rows[0][1][0][1].shape[:2]
ph = int(PANEL_W * h0 / w0)
LAB_H, PAD = 26, 6
canvas = Image.new("RGB",
                   (PAD + len(COLS) * (PANEL_W + PAD),
                    PAD + len(rows) * (ph + LAB_H + PAD)),
                   (255, 255, 255))
draw = ImageDraw.Draw(canvas)

for r, (stem, panels) in enumerate(rows):
    y = PAD + r * (ph + LAB_H + PAD)
    for c, (label, vis, n) in enumerate(panels):
        x = PAD + c * (PANEL_W + PAD)
        if vis is None:
            draw.text((x + 4, y + 4), f"{label} (n/a)", fill=(150, 150, 150), font=FONT)
            continue
        draw.text((x + 4, y + 4), f"{label}  |  {n} boxes", fill=(30, 30, 30), font=FONT)
        # 保持长宽比，居中留白（ExDark 混有 4:3 与 3:2，直接拉伸会变形）
        im = Image.fromarray(vis)
        sc = min(PANEL_W / im.width, ph / im.height)
        im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
        canvas.paste(im, (x + (PANEL_W - im.width) // 2,
                          y + LAB_H + (ph - im.height) // 2))
        draw.rectangle([x, y + LAB_H, x + PANEL_W - 1, y + LAB_H + ph - 1],
                       outline=(200, 200, 200))

path = OUT / "fig4_qualitative.png"
canvas.save(path, dpi=(300, 300))
print(f"\n已保存 -> {path}  ({canvas.size[0]}×{canvas.size[1]})")
