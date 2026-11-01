INDNet checkpoints — README
===========================

Companion to:
  "Beyond PSNR: A Two-Dataset Analysis of Low-Light Image Enhancement for Object Detection"
  Code:     https://github.com/1347993373/INDNet-LLIE-Detection
  Zenodo:   https://doi.org/10.5281/zenodo.23857840

Contents
--------
  baseline_unet_best.pth    Residual U-Net, both auxiliary branches disabled   8.05 M   20.8556 dB
  plus_illum_best.pth       + illumination branch                              8.24 M   20.6287 dB
  full_indnet_best.pth      Full model (illumination + noise + reconstruction)  8.28 M   21.1257 dB

Parameter counts differ because a disabled branch is not constructed at all; the
full model is the 8.28 M figure quoted in the paper.

Each file is a dict saved with torch.save:

  {
    "model":      <state_dict>,
    "best_psnr":  <float, best validation PSNR on LOL-v1>,
    "epoch":      <int>,
    "cfg":        <the full YAML config used for training>
  }

The training checkpoints also carried the Adam moments ("optimizer"). Those were
removed before archiving -- they roughly tripled the size and are not needed to
reproduce any reported number.

Load with:

  import torch, yaml
  from models.indnet import build_model

  rec  = torch.load("full_indnet_best.pth", map_location="cpu", weights_only=False)
  cfg  = rec["cfg"]                       # rebuild the exact architecture
  net  = build_model(cfg)
  net.load_state_dict(rec["model"])
  net.eval()

Note: rec["cfg"] is authoritative. Do NOT hand-write the architecture config --
illum_scale, base_channels and the two branch switches must match, otherwise
load_state_dict raises a size mismatch.

Training recipe recorded in "cfg"
---------------------------------
  total_iter   60000          patch 128   batch 8
  base_channels 48            num_blocks 6
  optimizer Adam, cosine schedule, AMP on
  loss w_rec 1.0, w_recon_low 0.0, w_illum_smooth 0.005,
       w_noise_l1 0.02, w_refl 0.5, w_illum_sup 1.0

Reported metrics (15-image LOL-v1 test set)
------------------------------------------
  full_indnet    PSNR 21.1257 dB   SSIM 0.7777   (best epoch 479)
  plus_illum     PSNR 20.6287 dB                 (best epoch 359)
  baseline_unet  PSNR 20.8556 dB                 (best epoch 419)

Note that the three are within 0.5 dB of each other, i.e. inside the standard error
of a 15-image test set; the paper does not rank them by PSNR, and the detection
results (the actual subject of the paper) are what separate them.

Known property of these weights
-------------------------------
The illumination map L saturates at its lower bound and is spatially constant.
This is documented in the paper: the "decoupling" mechanism does not in fact
recover a meaningful illumination map for this architecture. The checkpoints are
released for reproducibility of the reported numbers, not as a working Retinex
decomposition.

License: MIT. Third-party checkpoints (Zero-DCE, Retinexformer) are NOT included
here; they are downloaded automatically by notebooks/external_enhancers.py from
their original repositories and remain under their own licenses.
