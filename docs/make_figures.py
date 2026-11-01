"""生成论文 Fig. 1–3（矢量 PDF + 300 dpi PNG）。

用法：
    PYTHONPATH=<venv site-packages> python docs/make_figures.py

数据全部来自本项目实测日志，与 docs/PAPER_DRAFT.md 的 Table 1 / Appendix A 一致。
Fig. 4 / Fig. 5 已在 Kaggle 上生成（fig4_qualitative.png / coco_dark_sanity.png），
本脚本不重复绘制。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,        # TrueType，期刊要求可嵌入字体
    "ps.fonttype": 42,
})

GREY = "#4d4d4d"
BLUE = "#2f6f9f"
ORANGE = "#c46210"
RED = "#b02418"


def _box(ax, x, y, w, h, text, fc="#f2f5f8", ec=GREY, fs=6.8, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        linewidth=0.8, edgecolor=ec, facecolor=fc, clip_on=False, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", color="#1a1a1a", linespacing=1.4)


def _arrow(ax, x1, y1, x2, y2, label=None, color=GREY, fs=6.2, lx=0.0, ly=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
        linewidth=0.8, color=color, shrinkA=1, shrinkB=1, clip_on=False, zorder=1))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center",
                va="center", fontsize=fs, color=color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none"))


# --------------------------------------------------------------------------- #
# Fig. 1  三个协议
# --------------------------------------------------------------------------- #
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    Y = [0.87, 0.63, 0.37, 0.11]
    H = 0.15
    W = 0.86
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.06, 1.06)
        ax.axis("off")

    for ax, t in zip(axes, ["(a) Protocol I: frozen",
                            "(b) Protocol II-A: adapted",
                            "(c) Protocol II-B: cross-evaluated"]):
        ax.text(0.5, 1.03, t, ha="center", va="bottom", fontsize=7.4, fontweight="bold")

    # ---- (a)
    ax = axes[0]
    _box(ax, 0.5, Y[0], W, H, "low-light input $I$")
    _box(ax, 0.5, Y[1], W, H, "enhancement $E$  →  $I'$")
    _box(ax, 0.5, Y[2], W, H, "YOLOv8n\nfrozen weights", fc="#e8eef4", bold=True)
    _box(ax, 0.5, Y[3], W, H, "mAP@0.5 on ExDark", fc="white")
    _arrow(ax, 0.5, Y[0] - H / 2, 0.5, Y[1] + H / 2)
    _arrow(ax, 0.5, Y[1] - H / 2, 0.5, Y[2] + H / 2, "no gradient", color=BLUE, lx=0.155)
    _arrow(ax, 0.5, Y[2] - H / 2, 0.5, Y[3] + H / 2, "inference", color=BLUE, lx=0.13)

    # ---- (b)
    ax = axes[1]
    _box(ax, 0.5, Y[0], W, H, "pipeline $p$\ntrain split only")
    _box(ax, 0.5, Y[1], W, H, "fine-tune detector\n40 epochs × 3 seeds",
         fc="#fdf0e2", ec=ORANGE)
    _box(ax, 0.5, Y[2], W, H, "detector $D_p$\nfitted to $p$", fc="#e8eef4", bold=True)
    _box(ax, 0.5, Y[3], W, H, "mAP on $p$ test  = $A_p$", fc="white")
    _arrow(ax, 0.5, Y[0] - H / 2, 0.5, Y[1] + H / 2, color=ORANGE)
    _arrow(ax, 0.5, Y[1] - H / 2, 0.5, Y[2] + H / 2, "trained", color=ORANGE, lx=0.09)
    _arrow(ax, 0.5, Y[2] - H / 2, 0.5, Y[3] + H / 2)

    # ---- (c)
    ax = axes[2]
    _box(ax, 0.5, Y[0], W, H, "raw train split only")
    _box(ax, 0.5, Y[1], W, H, "fine-tune detector\n40 epochs × 3 seeds",
         fc="#fdf0e2", ec=ORANGE)
    _box(ax, 0.5, Y[2], W, H, "detector $D_{\\mathrm{raw}}$\nno adaptation",
         fc="#e8eef4", bold=True)
    _box(ax, 0.5, Y[3], W, H, "mAP on $p$ test  = $B_p$", fc="white")
    _arrow(ax, 0.5, Y[0] - H / 2, 0.5, Y[1] + H / 2, color=ORANGE)
    _arrow(ax, 0.5, Y[1] - H / 2, 0.5, Y[2] + H / 2, "trained on raw",
           color=ORANGE, lx=0.145)
    _arrow(ax, 0.5, Y[2] - H / 2, 0.5, Y[3] + H / 2, "domain shift", color=RED, lx=0.135)

    fig.text(0.5, -0.02,
             r"domain shift $\;=\;B_p-A_p$          "
             r"quality $\;=\;A_p-A_{\mathrm{raw}}$",
             ha="center", va="top", fontsize=7.4)
    fig.savefig(OUT / "fig1_protocols.pdf")
    fig.savefig(OUT / "fig1_protocols.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 2  亮度 vs mAP
# --------------------------------------------------------------------------- #
def fig2():
    # label: (Y, mAP, group, dx, dy, ha)
    pts = [
        ("raw",           37.1, 0.5338, "raw",       -9,  0.014, "left"),
        ("CLAHE",         52.8, 0.5367, "classical",  6,  0.011, "left"),
        ("gamma",         77.5, 0.5008, "classical", -6, -0.011, "right"),
        ("Res-UNet",      80.5, 0.5134, "ours",      -6,  0.012, "right"),
        ("Zero-DCE",      86.1, 0.4873, "learned",   -6, -0.012, "right"),
        ("Retinexformer", 89.8, 0.5204, "learned",    8,  0.013, "left"),
        ("INDNet",        90.0, 0.4957, "ours",       8, -0.013, "left"),
        ("+Illum",        96.8, 0.4739, "ours",       9, -0.012, "left"),
        ("hist. eq.",    119.1, 0.4494, "classical",  6, -0.013, "left"),
        ("MSR",          160.3, 0.3531, "classical",  5, -0.011, "left"),
        ("SSR",          157.4, 0.3573, "classical", -5,  0.012, "right"),
    ]
    msrcr = (106.2, 0.2660)
    style = {
        "raw":       dict(marker="s", color="black", s=22),
        "classical": dict(marker="o", color=BLUE, s=20),
        "learned":   dict(marker="^", color=ORANGE, s=26),
        "ours":      dict(marker="D", color=RED, s=16),
    }

    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.plot([28, 172], [-1.61e-3 * 28 + 0.6281, -1.61e-3 * 172 + 0.6281],
            color=GREY, linewidth=0.9, zorder=1)

    for lbl, x, y, k, dx, dy, ha in pts:
        ax.scatter(x, y, zorder=3, edgecolor="white", linewidth=0.5, **style[k])
        ax.annotate(lbl, (x, y), xytext=(x + dx, y + dy), fontsize=6.0,
                    ha=ha, va="center", color="#222222")

    ax.scatter(*msrcr, marker="X", s=30, color="#7a7a7a", zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.annotate("MSRCR\n(excluded)", msrcr, xytext=(msrcr[0] + 8, msrcr[1] - 0.004),
                fontsize=6.0, ha="left", va="center", color="#555555")

    # Retinexformer 与 INDNet：同亮度、差 0.025
    ax.annotate("", xy=(95.8, 0.5204), xytext=(95.8, 0.4957),
                arrowprops=dict(arrowstyle="<->", linewidth=0.7, color=RED))
    ax.text(99.4, 0.5080, r"$\Delta = 0.025$", fontsize=6.0, color=RED, va="center")

    ax.set_xlabel("mean luminance  $Y$")
    ax.set_ylabel("mAP@0.5")
    ax.set_xlim(25, 178)
    ax.set_ylim(0.245, 0.575)
    ax.grid(alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)

    mk = {"raw": "s", "classical": "o", "learned": "^", "ours": "D"}
    cl = {"raw": "black", "classical": BLUE, "learned": ORANGE, "ours": RED}
    handles = [Line2D([], [], linestyle="none", marker=mk[k], color=cl[k],
                      markersize=4.2, markeredgecolor="white", markeredgewidth=0.4,
                      label=k) for k in ("raw", "classical", "learned", "ours")]
    ax.legend(handles=handles, fontsize=6.0, frameon=False, loc="lower left",
              handletextpad=0.25, borderpad=0.15, labelspacing=0.22)
    ax.text(0.97, 0.95, r"$r=-0.954$", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.4, color=GREY)

    fig.savefig(OUT / "fig2_brightness_vs_map.pdf")
    fig.savefig(OUT / "fig2_brightness_vs_map.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 3  分类型
# --------------------------------------------------------------------------- #
def fig3():
    classes = ["bicycle", "boat", "bottle", "bus", "car", "cat",
               "chair", "cup", "dog", "motorbike", "people", "table"]
    data = {
        "raw":           [0.6233, 0.3671, 0.4901, 0.7946, 0.6555, 0.5083,
                          0.4777, 0.4065, 0.5030, 0.5987, 0.7116, 0.2697],
        "CLAHE":         [0.6321, 0.3740, 0.4810, 0.7883, 0.6643, 0.5087,
                          0.4783, 0.3898, 0.5087, 0.6085, 0.7196, 0.2876],
        "Retinexformer": [0.6288, 0.3746, 0.4833, 0.7756, 0.6226, 0.4852,
                          0.4631, 0.3863, 0.4713, 0.5723, 0.7023, 0.2799],
        "INDNet":        [0.6153, 0.3751, 0.4574, 0.7307, 0.5697, 0.4401,
                          0.4541, 0.3678, 0.4232, 0.5763, 0.6728, 0.2662],
    }
    colors = {"raw": "#9a9a9a", "CLAHE": BLUE,
              "Retinexformer": ORANGE, "INDNet": RED}

    y = np.arange(len(classes))[::-1]
    h = 0.2
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for i, (k, v) in enumerate(data.items()):
        ax.barh(y + (1.5 - i) * h, v, height=h, color=colors[k], label=k,
                edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=6.4)
    ax.set_xlabel("mAP@0.5 (per class)")
    ax.set_xlim(0, 0.86)
    ax.grid(axis="x", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.2, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handlelength=0.9, columnspacing=1.0,
              handletextpad=0.3)

    fig.savefig(OUT / "fig3_per_class.pdf")
    fig.savefig(OUT / "fig3_per_class.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    for p in sorted(OUT.glob("*")):
        print(f"{p.name:32s} {p.stat().st_size / 1024:7.1f} KB")
