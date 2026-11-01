"""1 分钟诊断格：看清增强图到底长什么样。

**不训练、不评测**，只处理 SAMPLE_N 张（默认 12）ExDark 图，做两件事：

  1. 导出并排对比图 → /kaggle/working/detection/samples/
     每一行 6 格: 原图 | Gamma | CLAHE | 残差U-Net | ＋光照 | INDNet(完整)
  2. 打印每种输入的统计量 —— 这是关键：
       · 平均亮度          看是否真的提亮了
       · 高光饱和比例       >10% 说明过曝（检测器最怕这个）
       · 暗部压死比例
       · 色彩偏移 (R-B)    明显偏移说明有严重色偏
       · 亮度标准差        对比度是否保留

为什么需要它：检测表显示"增强反而伤检测"。要判断是**增强图本身坏了**，
还是**这个冻结检测器就是不喜欢增强图**，最直接的办法就是看一眼 + 看统计。
"""

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2 as _cv2
except Exception:                       # pragma: no cover
    _cv2 = None

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working/INDNet")
ABL = Path("/kaggle/working/ablations")
OUT = Path("/kaggle/working/detection/samples")
SAMPLE_N = int(os.environ.get("SAMPLE_N", "12"))
ENH_MAX_SIDE = int(os.environ.get("EXDARK_ENH_MAX_SIDE", "1024"))
PANEL_W = 320                           # 拼图里每格的显示宽度
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_ckpt(path):
    """兼容新旧 torch：2.6+ 把 weights_only 的默认值从 False 改成了 True。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def enhance_capped(net, t, cap):
    h, w = t.shape[-2:]
    if max(h, w) <= cap:
        return net.enhance(t)
    s = cap / max(h, w)
    sh, sw = max(16, int(round(h * s))), max(16, int(round(w * s)))
    small = F.interpolate(t, size=(sh, sw), mode="bilinear",
                          align_corners=False, antialias=True)
    out = net.enhance(small)
    del small
    return F.interpolate(out, size=(h, w), mode="bilinear",
                         align_corners=False, antialias=True).clamp(0.0, 1.0)


def classical_modes(rgb_u8):
    out = {"gamma": (np.power(rgb_u8 / 255.0, 0.5) * 255.0).round().astype(np.uint8)}
    if _cv2 is not None:
        lab = _cv2.cvtColor(rgb_u8, _cv2.COLOR_RGB2LAB)
        clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out["clahe"] = _cv2.cvtColor(lab, _cv2.COLOR_LAB2RGB)
    return out


def stats(name: str, rgb_u8: np.ndarray) -> dict:
    a = rgb_u8.astype(np.float32)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    ym = float(y.mean()) + 1e-6
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = float(((mx - mn) / np.maximum(mx, 1e-6)).mean() * 100)
    r_m, b_m = float(a[..., 0].mean()), float(a[..., 2].mean())
    return {
        "name": name,
        "mean": float(y.mean()),
        "sat_hi": float((y > 250).mean() * 100),
        "crush_lo": float((y < 5).mean() * 100),
        # 【重要】绝对 R−B 会随亮度同比放大，不能直接跨方法比较！
        # 提亮 2.6 倍，绝对色差自然也放大 2.6 倍 —— 那不是偏色。
        # 判断"有没有偏色"必须看按亮度归一后的相对色偏。
        "cast_abs": r_m - b_m,
        "cast_rel": (r_m - b_m) / ym * 100,
        "sat": sat,
        "std": float(y.std()),
    }


def restore_weights() -> list[str]:
    """/kaggle/working 被清空时，从挂载的 Input 里找回权重。"""
    all_ck = sorted(IN.rglob("best.pth"), key=lambda p: len(str(p))) if IN.exists() else []
    restored = []
    for name in ("baseline_unet", "plus_illum", "full_indnet"):
        dst = ABL / name / "best.pth"
        if dst.is_file():
            continue
        for c in all_ck:
            if c.parent.name == name:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(c.read_bytes())
                restored.append(name)
                break
    return restored


def main() -> None:
    import sys

    # ---------- 代码 ----------
    roots = []
    if IN.exists():
        for d in IN.rglob("*"):
            if d.is_dir() and (d / "train.py").is_file() and (d / "models" / "indnet.py").is_file():
                roots.append(d)
    roots = sorted(set(roots), key=lambda p: len(str(p)))
    roots = [p for p in roots if "INDNet" not in str(p)] or roots
    if not roots:
        log("[!] 找不到代码包（需含 train.py 与 models/indnet.py）")
        return
    WORK.parent.mkdir(parents=True, exist_ok=True)
    if not (WORK / "models" / "indnet.py").is_file():
        import shutil

        if WORK.exists():
            shutil.rmtree(WORK)
        shutil.copytree(roots[0], WORK)
    os.chdir(WORK)
    sys.path.insert(0, str(WORK))
    from models.indnet import INDNet

    restored = restore_weights()
    log(f"权重恢复: {restored if restored else '（/kaggle/working 里已有）'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device = {device}")

    # ---------- 增强模型 ----------
    enhancers = {}
    for name, (fi, fn) in (("baseline_unet", (False, False)),
                           ("plus_illum", (True, False)),
                           ("full_indnet", (True, True))):
        ck = ABL / name / "best.pth"
        if not ck.is_file():
            log(f"  [跳过] 缺少 {ck}")
            continue
        rec = load_ckpt(ck)
        m = (rec.get("cfg") or {}).get("model", {})
        net = INDNet(in_channels=m.get("in_channels", 3),
                     base_channels=m.get("base_channels", 32),
                     num_blocks=m.get("num_blocks", 4),
                     use_illum=m.get("illumination_branch", fi),
                     use_noise=m.get("noise_branch", fn))
        net.load_state_dict(rec["model"])
        net.to(device).eval()
        enhancers[name] = net
    log(f"已载入增强模型: {list(enhancers)}")

    # ---------- 找图 ----------
    mount = None
    for p in IN.rglob("*"):
        if p.is_dir() and "exdark" in p.name.lower():
            mount = p if mount is None or len(str(p)) < len(str(mount)) else mount
    if mount is None:
        log("[!] 没找到 ExDark 数据集")
        return
    best, n_img = None, 0
    for d in mount.rglob("*"):
        if d.is_dir():
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
            if n > n_img:
                best, n_img = d, n
    imgs = sorted(f for f in best.iterdir() if f.suffix.lower() in IMG_EXT)
    log(f"图像目录 {best}（{n_img} 张）")

    # 等距抽样，保证覆盖不同尺寸
    step = max(1, len(imgs) // SAMPLE_N)
    picks = imgs[::step][:SAMPLE_N]
    OUT.mkdir(parents=True, exist_ok=True)

    order = ["original", "gamma", "clahe", "baseline_unet", "plus_illum", "full_indnet"]
    labels = {"original": "原图", "gamma": "Gamma", "clahe": "CLAHE",
              "baseline_unet": "残差U-Net", "plus_illum": "+光照", "full_indnet": "INDNet完整"}
    all_stats: dict[str, list] = {k: [] for k in order}

    log()
    log("=" * 92)
    log("  每种输入的统计（全样本平均）")
    log("=" * 92)

    for i, p in enumerate(picks, 1):
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        modes = {"original": rgb}
        modes.update(classical_modes(rgb))

        t = torch.from_numpy(rgb.astype(np.float32) / 255.0
                             ).permute(2, 0, 1).unsqueeze(0).to(device)
        if enhancers:
            with torch.no_grad():
                for name, net in enhancers.items():
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    out = enhance_capped(net, t, ENH_MAX_SIDE)
                    modes[name] = (out[0].permute(1, 2, 0).cpu().numpy() * 255
                                   ).round().astype(np.uint8)
                    del out
        del t

        # 统计
        for k in order:
            if k in modes:
                all_stats[k].append(stats(k, modes[k]))

        # 拼图
        panels = []
        for k in order:
            if k not in modes:
                continue
            im = Image.fromarray(modes[k])
            ph = max(1, int(round(PANEL_W * im.height / im.width)))
            im = im.resize((PANEL_W, ph), Image.BILINEAR)
            panels.append((labels[k], im))
        ph = max(im.height for _, im in panels)
        canvas = Image.new("RGB", (PANEL_W * len(panels), ph + 18), (255, 255, 255))
        for j, (lab, im) in enumerate(panels):
            canvas.paste(im, (j * PANEL_W, 18))
        canvas.save(OUT / f"cmp_{i:02d}_{p.stem}.png")
        log(f"  [{i}/{len(picks)}] {p.name}  {w}×{h}  -> cmp_{i:02d}_{p.stem}.png")

    log()
    hdr = (f"{'输入':<14}{'平均亮度':>10}{'提亮倍数':>10}{'高光饱和%':>11}"
           f"{'暗部压死%':>11}{'相对色偏%':>11}{'饱和度%':>9}{'亮度std':>9}")
    log(hdr)
    log("-" * 100)
    base_mean = None
    for k in order:
        s = all_stats[k]
        if not s:
            continue
        m = float(np.mean([x["mean"] for x in s]))
        if k == "original":
            base_mean = m
        ratio = (m / base_mean) if base_mean else float("nan")
        log(f"{labels[k]:<14}"
            f"{m:>10.1f}"
            f"{ratio:>9.2f}x"
            f"{np.mean([x['sat_hi'] for x in s]):>11.2f}"
            f"{np.mean([x['crush_lo'] for x in s]):>11.2f}"
            f"{np.mean([x['cast_rel'] for x in s]):>11.1f}"
            f"{np.mean([x['sat'] for x in s]):>9.1f}"
            f"{np.mean([x['std'] for x in s]):>9.1f}")

    log()
    log(f"对比图已保存 -> {OUT}（共 {len(picks)} 张，每张并排）")
    log("判读要点：")
    log("  · 高光饱和% > 10 → 过曝，检测器最怕这个（最可疑的掉点原因）")
    log("  · 提亮倍数：原图均值 44.8 属暗场，正常曝光约 100~130，")
    log("    所以 2.5~3x 本身不算错，但要结合过曝一起看")
    log("  · 相对色偏% = (R均值−B均值)/亮度 —— 【必须用这个比色偏】。")
    log("    绝对 R−B 会随亮度同比放大，提亮 2.6 倍时它自然也大 2.6 倍，那不是偏色。")
    log("  · 饱和度% 与亮度 std：明显偏离原图说明色彩/对比被扭曲")


main()
