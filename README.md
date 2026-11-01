# Beyond PSNR: A Two-Dataset Analysis of Low-Light Image Enhancement for Object Detection

Code, evaluation pipeline and figures for the paper:

> **Beyond PSNR: A Two-Dataset Analysis of Low-Light Image Enhancement for Object Detection**
> *Multimedia Tools and Applications* (submitted)

This repository is **not** a "new SOTA enhancer". It is the full experimental apparatus behind a
controlled study of a routine assumption: *that enhancing a low-light image before a detector must help*.
We answer it with 23 condition–recipe rows, two datasets, two protocols and two detectors.

---

## Main findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **No enhancement reliably improves detection.** Of 12 inputs on ExDark, only CLAHE is neutral (+0.003 mAP); the other 11 degrade it, from −0.013 to −0.268, including Retinexformer (ICCV'23). | Table 1 |
| 2 | **PSNR and SSIM do not predict detection.** On COCO with a retained ground truth, CLAHE improves PSNR by **+2.82 dB** and SSIM by **+0.166** over the darkened input, yet **lowers** mAP by 0.0201 (3.3× sampling noise). | Table 7 |
| 3 | **The loss decomposes into noise, restoration error and domain shift.** Sensor noise accounts for 58% of the drop; after re-training a detector per pipeline, 80% of the residual loss of a learned enhancer is domain shift, not image quality. | §6.2, §6.3, Table 4 |
| 4 | **Brightness is a within-dataset proxy, not a law.** Mean luminance explains 85% of the mAP variance on ExDark (r = −0.954) but the sign flips on COCO — we report this failed transfer rather than hiding it. | §4.2, Fig. 2 |

Measurements are deterministic: re-running the full pipeline reproduces every mAP to four decimals.

---

## Repository layout

```
INDNet/
├── train.py, test.py                 INDNet training / evaluation entry points
├── models/indnet.py                  Residual restoration backbone (4-level U-Net + SE, 8.28 M)
├── losses/losses.py                  Reconstruction + illumination + noise regularisers
├── datasets/lol_dataset.py           Paired low-light dataset (auto-detects directory layout)
├── utils/{common,metrics}.py         Config, logging, seeds, PSNR / SSIM / LPIPS
├── configs/                          lol_v1 (local) and lol_v1_kaggle (mounted paths)
├── notebooks/                        ★ the actual experimental pipeline
│   ├── kaggle_pipeline.py            single entry point: preflight → weights → detection → A2
│   ├── kaggle_exdark_detect_cell.py  12-input frozen-detector evaluation (Table 1, 2)
│   ├── kaggle_exdark_finetune_cell.py  Protocol II-A / II-B, multi-seed (Table 4)
│   ├── kaggle_coco_cell.py           COCO + synthetic degradation + fidelity (Tables 6, 7)
│   ├── kaggle_figure4_cell.py        qualitative panels (Fig. 4)
│   ├── external_enhancers.py         Zero-DCE and Retinexformer weight loading
│   └── kaggle_preflight.py           2-minute environment check
├── docs/
│   ├── PAPER_DRAFT.md                full manuscript source (markdown)
│   ├── TABLES.tex                    all tables, booktabs (generated)
│   ├── STATEMENTS.md                 declarations + cover letter
│   ├── make_figures.py               regenerates Fig. 1–3 (300 dpi PNG + vector PDF)
│   ├── md_tables_to_tex.py           regenerates TABLES.tex from PAPER_DRAFT.md
│   ├── figures/                      Fig. 1–3 (Fig. 4–5 come from the Kaggle runs)
│   └── KAGGLE_GUIDE.md               operational notes for the Kaggle environment
├── scripts/pack_for_kaggle.py        bundles the code for the Kaggle dataset
└── requirements.txt                  local dependencies
```

`data/`, `experiments/`, `results/`, `runs/` and all `*.pth` are excluded — see *Data & weights* below.

---

## Environment

```bash
python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # Linux / macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

On Kaggle, **do not** run `pip install -r requirements.txt` — the image ships a CUDA-matched
torch and reinstalling from PyPI replaces it with the CPU build. Install only `ultralytics`
and `einops`, as `notebooks/kaggle_preflight.py` does.

---

## Reproducing the results

All experiments run on a single Tesla T4 through Kaggle Notebooks. The pipeline is invoked from one
cell; behaviour is controlled by environment variables.

### Table 1 / Table 2 — 12 inputs, frozen YOLOv8n, ExDark

```python
import os, glob
os.environ["SKIP_ABLATION"]          = "1"
os.environ["EXDARK_CLASSICAL_MODES"] = "gamma,clahe,histeq,ssr,msr,msrcr"
os.environ["EXDARK_EXTRA"]           = "zero_dce,retinexformer"
os.environ["EXDARK_MODEL"]           = "yolov8n.pt"
os.environ["EXDARK_MODES"]           = ("original,gamma,clahe,histeq,ssr,msr,msrcr,"
                                        "zero_dce,retinexformer,baseline_unet,plus_illum,full_indnet")

p = sorted(glob.glob("/kaggle/input/**/kaggle_pipeline.py", recursive=True), key=len)[0]
exec(open(p, encoding="utf-8").read())
```

Expect `mAP@0.5` = 0.5338 (raw) … 0.5367 (CLAHE) … 0.5204 (Retinexformer). Runtime ≈ 4 h.

### Table 4 — Protocol II-A / II-B, three seeds

```python
os.environ.update(SKIP_ABLATION="1", SKIP_DETECT="1", RUN_FINETUNE="1",
                  FT_MODES="original,clahe,full_indnet",
                  FT_SEEDS="3407,1234,2024", FT_EPOCHS="40", FT_CLEAN_TRAIN="1")
```

Expect 0.6818 ± 0.0021 (raw), 0.6859 ± 0.0026 (CLAHE), 0.6698 ± 0.0036 (INDNet) under II-A.
The split seed is fixed at 3407 so the reported std reflects **training** variance only.

### Tables 6 / 7 — COCO val2017 with a controlled degradation

```python
os.environ["COCO_CONDS"] = ("original,original_clahe,dark,dark_clahe,"
                            "dark_indnet,dark_retinexformer")
# noise ablation: add  os.environ["COCO_DARK_NOISE"] = "0"
```

Expect 0.5700 / 0.5629 / 0.4905 / 0.4704 / 0.4840 / 0.5025 (noisy recipe).

### Figures and tables

```bash
python docs/make_figures.py        # Fig. 1 2 3  → docs/figures/*.pdf, *.png
python docs/md_tables_to_tex.py    # Table 1–7, A1 → docs/TABLES.tex
```

---

## Data & weights

* **ExDark** — `hongngnguyn/exdarkdataset` on Kaggle (7,363 images, 23,710 boxes, 12 classes).
* **COCO val2017 subset** — `bardiaardakanian/mmsample` (5,000 images + `instances_val2017.json`).
* **LOL-v1** — `soumikrakshit/lol-dataset`, used only for the PSNR/SSIM reference numbers.

Datasets are **not** redistributed here; please obtain them from the sources above.

Trained weights (`baseline_unet`, `plus_illum`, `full_indnet`) and the COCO/ExDark result files are
archived on Zenodo: **https://doi.org/10.5281/zenodo.23857840** — the concept DOI, which always
resolves to the latest version (the v1.0.0 snapshot is `10.5281/zenodo.23857841`).

---

## Notes on the enhancement models we compare against

* **Zero-DCE** — official CVPR'20 checkpoint.
* **Retinexformer** — ICCV'23 checkpoint, loaded with the correct hyper-parameters
  (`n_feat=40, stage=1, num_blocks=[1,2,2]`), which differ from the repository defaults.
* All classical operators use **out-of-the-box parameters** (CLAHE: `clipLimit=2.0`, `8×8` tiles;
  γ = 0.5). No operator is tuned on the evaluation domain — tuning there would leak test information
  into the pipeline. Our numbers should therefore be read as *untuned* performance.

---

## Citation

```bibtex
@article{du2026beyondpsnr,
  title   = {Beyond PSNR: A Two-Dataset Analysis of Low-Light Image Enhancement
             for Object Detection},
  author  = {Du, Jinfeng},
  journal = {Multimedia Tools and Applications},
  year    = {2026},
  note    = {Code and data: \url{https://doi.org/10.5281/zenodo.23857840}}
}

@software{du2026indnetcode,
  author  = {Du, Jinfeng},
  title   = {Code for "Beyond PSNR: A Two-Dataset Analysis of Low-Light Image
             Enhancement for Object Detection"},
  year    = {2026},
  doi     = {10.5281/zenodo.23857840},
  url     = {https://github.com/1347993373/INDNet-LLIE-Detection}
}
```

## License

MIT (see `LICENSE`). The bundled third-party checkpoints (Zero-DCE, Retinexformer) remain under
their original licenses.
