"""Bundle the three ablation checkpoints into weights.zip for Zenodo.

Why this script exists
----------------------
Kaggle's Output download hands you files named ``best.zip`` / ``best (1).zip`` --
but those are NOT archives: a PyTorch checkpoint saved with the zip serialisation
format *is itself* a zip file, so the browser just appended ``.zip``. The content
is the checkpoint, and ``torch.load`` reads it fine regardless of the extension.

All three variants are also called ``best`` internally, so the only reliable way
to tell them apart is the ``cfg.model`` branch switches stored inside the
checkpoint. That is what this script does: it loads every candidate, decides the
variant, verifies the parameter count and best PSNR, and writes a clean zip.

All console output is ASCII on purpose: printing Chinese into Git Bash on a GBK
codepage garbles the terminal.

Usage
-----
    python scripts/pack_weights.py --src "C:/Users/DES/Desktop/best"

Options
-------
    --out            output path, default <project>/weights.zip
    --allow-missing  package a subset instead of requiring all three
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

import torch

ROOT = Path(__file__).resolve().parent.parent

# (illumination_branch, noise_branch) -> variant name
VARIANT = {
    (False, False): "baseline_unet",
    (True, False): "plus_illum",
    (True, True): "full_indnet",
}
ORDER = ["baseline_unet", "plus_illum", "full_indnet"]
SUFFIXES = (".pth", ".pt", ".pth.tar", ".zip", ".bin")


def _load(obj):
    try:
        return torch.load(obj, map_location="cpu", weights_only=False)
    except TypeError:                      # torch < 1.13
        return torch.load(obj, map_location="cpu")


def _is_ckpt(obj) -> bool:
    return isinstance(obj, dict) and "model" in obj


def iter_candidates(src: Path) -> Iterator[tuple[str, object]]:
    """Yield (label, checkpoint).

    Strategy: try torch.load on the file itself first -- that covers both a bare
    .pth and a checkpoint whose zip serialisation merely *looks* like an archive.
    Only if that fails do we treat the file as a real archive of checkpoints.
    """
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUFFIXES:
            continue
        try:
            obj = _load(p)
            if _is_ckpt(obj):
                yield p.name, obj
                continue
        except Exception:                  # noqa: BLE001  - fall through to archive mode
            pass

        if p.suffix.lower() != ".zip":
            print(f"  [skip] {p.name}: not a project checkpoint")
            continue
        try:
            with zipfile.ZipFile(p) as z:
                members = [i.filename for i in z.infolist()
                           if i.filename.lower().endswith((".pth", ".pt", ".bin"))
                           and not i.is_dir()]
        except zipfile.BadZipFile:
            print(f"  [skip] {p.name}: not a zip and not a checkpoint")
            continue
        if not members:
            print(f"  [skip] {p.name}: archive holds no checkpoint")
            continue
        for m in members:
            label = f"{p.name} > {m}" if len(members) > 1 else p.name
            try:
                with zipfile.ZipFile(p) as z:
                    obj = _load(io.BytesIO(z.read(m)))
                if _is_ckpt(obj):
                    yield label, obj
                else:
                    print(f"  [skip] {label}: not a project checkpoint")
            except Exception as exc:       # noqa: BLE001
                print(f"  [skip] {label}: unreadable ({type(exc).__name__})")


def classify(rec: dict) -> tuple[str, int, float, int]:
    cfg = rec.get("cfg") or {}
    mc = cfg.get("model") or {}
    if "illumination_branch" not in mc:
        raise ValueError("checkpoint has no model config, cannot identify the variant")
    key = (bool(mc["illumination_branch"]), bool(mc["noise_branch"]))
    if key not in VARIANT:
        raise ValueError(f"unknown branch combination {key}")
    n = sum(v.numel() for v in rec["model"].values() if hasattr(v, "numel"))
    return (VARIANT[key], n, float(rec.get("best_psnr", float("nan"))),
            int(rec.get("epoch", -1) or -1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder holding best.zip / best.pth (recursive)")
    ap.add_argument("--out", default=str(ROOT / "weights.zip"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()

    src = Path(a.src).expanduser()
    if not src.is_dir():
        raise SystemExit(f"[error] folder not found: {src}")

    print(f"Scanning {src}\n")
    found: dict[str, tuple[str, dict, int, float, int]] = {}
    seen = 0
    for label, rec in iter_candidates(src):
        seen += 1
        try:
            name, n, psnr, epoch = classify(rec)
        except ValueError as exc:
            print(f"  [skip] {label}: {exc}")
            continue
        if name in found:
            print(f"  [dup ] {label} is also {name}; keeping {found[name][0]}")
            continue
        found[name] = (label, rec, n, psnr, epoch)
        print(f"  [OK  ] {label:<16} -> {name:<14} params {n/1e6:6.2f}M  "
              f"best_psnr {psnr:7.4f}  epoch {epoch}")

    if seen == 0:
        raise SystemExit("no checkpoint found -- point --src at the folder holding them")

    missing = [v for v in ORDER if v not in found]
    if missing:
        line = "missing variant(s): " + ", ".join(missing)
        if not a.allow_missing:
            raise SystemExit(f"\n{line} -- download all three, or pass --allow-missing")
        print(f"\n[warn] {line} (continuing because --allow-missing)")

    readme = ROOT / "docs" / "weights_README.txt"
    out = Path(a.out).expanduser()
    if out.exists():
        out.unlink()

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        for name, (_label, rec, _n, _p, _e) in found.items():
            # Keep only what inference needs. The training checkpoints also carry the
            # Adam moments (~2x the model size); dropping them roughly thirds the zip
            # and changes nothing for reproducing the reported numbers.
            slim = {k: rec[k] for k in ("model", "cfg", "best_psnr", "epoch")
                    if k in rec}
            torch.save(slim, stage / f"{name}_best.pth")
        if readme.is_file():
            (stage / "README.txt").write_bytes(readme.read_bytes())
        else:
            print(f"\n[warn] {readme} not found; the zip will carry no README")

        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in sorted(stage.iterdir()):
                z.write(f, arcname=f.name)

    print(f"\nWrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    with zipfile.ZipFile(out) as z:
        for info in z.infolist():
            print(f"  {info.filename:<28} {info.file_size/1e6:8.2f} MB")
    print("\nNext: Zenodo record -> New version -> Upload files -> Publish")
    print("(creates v1.0.1; concept DOI 10.5281/zenodo.23857840 stays the same)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
