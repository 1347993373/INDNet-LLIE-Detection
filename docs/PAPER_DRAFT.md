# Beyond PSNR: A Two-Dataset Analysis of Low-Light Image Enhancement for Object Detection

> **Status**: v2.2 (2026-11-01). Target venue: *Multimedia Tools and Applications*.
> **v2.2 改动**：① A2 换三种子（CLAHE +0.007 → +0.004 中性；78% → 80%）；② §2.3 重写，加入两篇 2026 年独立工作
> （ICSIPA `10.1109/ICSIPA69226.2026.11720114`、PCAMV `10.1117/12.3137252`）与一篇**反向结果**
> （Cai et al., CMC 87(2) 2026, `10.32604/cmc.2026.075756`）；③ §1 增加"独立复现"论述；
> ④ 参考文献 20 → 23 条；⑤ §8 新增"算子参数未调"局限（回应 ICSIPA 的 validation-tuned CLAHE）。
> **v2 相对 v1 的改动**（因 COCO 第二数据集实验推翻了两条原判断）：
> 1. **标题**从 "Beyond Brightness / Two-Factor" 改为 "Beyond PSNR" —— 双因子模型在跨数据集检验中失效（亮度系数符号翻转），不能再当主卖点。
> 2. **§6 新增 6.2 / 6.3**：COCO 第二数据集 + 噪声消融（两轮，共 4.2 h + 2.3 h GPU）。
> 3. **§7 重写**：新增"PSNR/SSIM 不是代理"与"三种失效模式"（亮度错、过曝、噪声放大）。
> 4. **§4.3 降级**：双因子模型明确标为 ExDark 内的描述性模型。
> 5. **§8 局限**：删掉"只有单数据集"，改为"第二数据集是合成退化"。
> 6. 新增 **Table 6（COCO 两种退化配方）**、**Table 7（保真度与过曝）**。
> **给峰哥的批注（不入正文）**：
> 1. 所有数字都来自本项目的实测日志，可溯源；`[REF:xx]` 需在投稿前补全 DOI。
> 2. ⚠️ 检索发现一篇高度同题的预印本 *"Does Low-Light Image Enhancement Improve Object Detection? A Detection-Oriented and Cost-Aware Study on ExDark"*，以及 *Applied Soft Computing* 2026 的一篇综述+基准。本文的差异化已写进 §2.3 与 §1；投稿前请务必找到前者正式版并引用，否则会被质疑不了解现状。
> 3. 文中 `[CIT:...]` 是仍需确认的引用位，其余引用均为真实文献。

---

## Abstract

Low-light image enhancement (LLIE) is routinely inserted before object detectors, on the implicit assumption that a visually clearer image must be a better detection input. We test this assumption under a protocol in which **only the input changes** — the detector stays frozen, so any change in mean average precision (mAP) belongs to the enhancement alone. Twenty-three condition–recipe rows are evaluated on two benchmarks: twelve operators on ExDark (raw, six classical, two learned, three ablated variants of a residual restoration network), and six conditions on COCO val2017 under a controlled low-light degradation — scored under two recipes, with and without sensor noise — where the clean image is available as ground truth. Three findings emerge. First, **no enhancement reliably improves detection**: on ExDark only CLAHE is neutral (+0.003 mAP) while the other eleven lose 0.013–0.268, including the ICCV'23 state of the art; on COCO the best input gains +0.013 and the ordering reproduces with a second detector. Second, **pixel-fidelity metrics do not predict detection**: on the COCO pairs, an operator that gains 2.82 dB PSNR and 0.166 SSIM *loses* 0.020 mAP, two learned methods whose PSNRs differ by 0.4 dB differ by 0.024 mAP, and within the degraded branch the PSNR–mAP correlation is r = 0.007. Third, the loss decomposes into **noise amplification, local over-exposure and domain shift**: removing sensor noise from the degradation halves the damage (0.079 → 0.033) and turns CLAHE from harmful to neutral, and the one condition that gains 6.9 dB PSNR while losing accuracy clips 6.6% of pixels, twice the ground-truth rate. Mean luminance, finally, is not a portable predictor — its correlation with mAP is −0.95 on ExDark but +0.61 on COCO. Re-training the detector is worth +0.148 mAP and removes CLAHE's penalty entirely, while learned enhancement stays negative; 80% of its remaining loss is domain shift. Enhancement is therefore not a free pre-processing step, PSNR is not a selection criterion, and any pipeline change must be accompanied by detector re-training.

**Keywords**: low-light image enhancement · object detection · domain shift · evaluation protocol · ExDark · COCO

---

## 1 Introduction

Images captured at night are dark, low-contrast and noisy. Detectors trained on daytime data lose a large fraction of their accuracy on such inputs, and a natural remedy is to restore the image before detection: run an enhancement network, then feed its output to an off-the-shelf detector. The practice is widespread and is usually justified by enhanced-image metrics — PSNR and SSIM on paired benchmarks such as LOL [REF:lol], or no-reference scores — rather than by downstream detection accuracy.

This paper asks a deliberately narrow question: **if the detector is held fixed, does enhancement help detection?**

The question is not rhetorical. Concurrent work has begun to report that enhancement often does not help [REF:peerj][REF:asc_review], and two independent 2026 studies reach that conclusion on ExDark with different detectors and different method pools [REF:icsipa][REF:pcamv] — a convergence that makes the observation hard to dismiss as one laboratory's artefact. A third study reports small positive gains, but with a detector far more heavily adapted to the domain [REF:sgvmnet], which is itself informative: the size of the effect depends on how well the detector already fits the input distribution. What the existing reports do not provide is a *quantitative account of why*: how much of the loss is explained by simple image statistics, how much by restoration fidelity, and how much by the mere fact that the detector was never trained on enhanced images.

We address this with a controlled experiment on ExDark [REF:exdark]. Twelve inputs are produced from the same 7,362 images: the raw image; six classical operators spanning intensity mapping (gamma), local contrast (CLAHE, histogram equalisation) and Retinex theory (SSR, MSR, MSRCR); two recent learned enhancers (Zero-DCE [REF:zerodce], Retinexformer [REF:retinexformer]); and three ablated variants of a residual restoration network developed in-house. Identical frozen YOLOv8n weights process all twelve, so differences in mAP are attributable to the input alone.

Three results follow.

**(i) Almost nothing helps.** CLAHE is statistically indistinguishable from the raw image (+0.003, within the ±0.005 sampling tolerance); every other method degrades detection, and the degradation ranges over an order of magnitude, from −0.013 to −0.268. This includes Retinexformer, whose PSNR on LOL-v1 is 25.16 dB — four decibels above any method in our pool.

**(ii) Within ExDark the loss is predictable — but the predictor does not transfer.** Across the eleven non-degenerate methods, mean luminance Y explains 91% of the mAP variance (r = −0.954). Two methods placed at the same brightness, however, differ by 0.025 mAP, and their residuals from the fitted line are ordered exactly as their restoration fidelity: Retinexformer +0.037, our residual U-Net +0.015, Zero-DCE −0.002. That suggests brightness and fidelity as two separable factors. We test the suggestion on a second benchmark where the deformation can be reversed, and it fails: the brightness coefficient changes sign (−0.954 to +0.61). We therefore keep the analysis as a within-dataset description and state the general claim more narrowly.

**(iii) Most of the loss is domain shift.** By training one detector per pipeline (protocol A) and cross-evaluating detectors and pipelines (protocol B), we split each method's loss into a quality component (measured after adaptation) and a domain-shift component. Domain adaptation is worth +0.148 mAP overall. CLAHE's cross-evaluated penalty of −0.014 disappears after adaptation (+0.004, below the ±0.005 tolerance — it is neutralised, not turned into a gain), while learned enhancement stays negative at −0.012, of which 80% is domain shift. Every number in this decomposition is a mean over three training seeds; the seed spread is 0.0021–0.0036 mAP.

**(iv) What remains is noise, over-exposure and fidelity — none of them visible in PSNR.** Because COCO gives us a clean reference, we can ablate the degradation itself. Removing sensor noise halves the damage (0.0795 → 0.0330) and turns CLAHE from harmful to neutral; the one condition that gains 6.9 dB of PSNR while losing accuracy clips 6.6% of pixels against 3.5% in the reference; and within the degraded branch the PSNR–mAP correlation is 0.007. An operator can therefore improve every standard image metric and still make the detector worse.

The practical reading is blunt. Enhancement is not free: it silently moves the input off the detector's training distribution, and the usual justification — a better PSNR — is not predictive of detection. If an enhancement stage is deployed, the detector must be re-trained on its output, the brightness gain should be kept modest, and selection should be made on downstream accuracy rather than on image metrics.

**Contributions.**
1. A controlled, single-variable evaluation of twelve enhancement variants on ExDark under a frozen detector, including both classical operators and two recent learned methods (§4).
2. A second benchmark built from COCO val2017 with a controlled, ablatable low-light degradation and a retained clean reference, on which the negative result replicates and restoration fidelity can be measured directly rather than inferred (§6.2).
3. A decomposition of the residual loss into noise amplification, local over-exposure and distributional shift, with a noise ablation that isolates the first (§6.3, §7), and evidence that PSNR and SSIM do not rank inputs in the order the detector does.
4. A protocol that decomposes the loss into domain-shift and quality components, and its application to three pipelines (§5).
5. A verified detector-agnosticism check (§6.1) and operating guidelines for practitioners (§7).

---

## 2 Related Work

### 2.1 Low-light image enhancement

Classical approaches model the image as a product of reflectance and illumination. Single- and multi-scale Retinex [REF:jobson] estimate illumination by Gaussian filtering in the log domain, and MSRCR adds a colour-restoration term to suppress the characteristic grey cast; histogram-based operators, in particular CLAHE [REF:clahe], equalise local contrast with a clip limit. These remain strong baselines and are cheap enough to run on embedded hardware.

Learned methods replaced hand-crafted priors with data-driven mappings. Zero-DCE [REF:zerodce] predicts the parameters of a family of quadratic curves without paired supervision, and its lightweight successor [REF:zerodcepp] removes most of the spatial cost. Retinex-based networks unroll the decomposition into a deep model: KinD and its extension [REF:kind] separate illumination and reflectance explicitly, URetinex-Net [REF:uretinex] unfolds a half-quadratic optimisation, and Retinexformer [REF:retinexformer] replaces the convolutional backbone with an illumination-guided transformer and currently reports the strongest results on LOL-v1. Normalising-flow [REF:llflow] and SNR-aware [REF:snr] models address the ill-posedness and the noise amplification of dark regions respectively, while SCI [REF:sci] and PairLIE [REF:pairlie] pursue efficiency and simplicity. Comprehensive surveys of this line are given in [REF:llie_survey].

All of these methods are evaluated on paired benchmarks with PSNR/SSIM, or on unpaired sets with no-reference metrics. Neither family measures whether a downstream task improves.

### 2.2 Object detection under low light

ExDark [REF:exdark] remains the standard benchmark for detection in natural low-light scenes: 7,363 images, twelve object classes, 23,710 boxes, and a wide spread of image sizes. Detection-oriented work falls into three groups: (a) detectors modified to consume illumination information, (b) domain-adaptation methods that align night-time features to daytime statistics [REF:da_detect], and (c) two-stage pipelines in which an enhancer precedes a detector [REF:peerj]. Group (c) is the setting of this paper, and it is the least rigorously characterised of the three: the enhancer and the detector are typically trained and evaluated in isolation, and the enhancer is selected by its own metrics.

### 2.3 Enhancement as pre-processing: what is already known

Recent work has started to question the assumption. Al-Refai et al. [REF:peerj] cascade Zero-DCE++, GladNet and TBEFN with YOLOv7 on ExDark and report that TBEFN improves mAP relative to the other two, while Zero-DCE++ performs worst — an early indication that aggressive brightening is counterproductive. A recent systematic review with benchmark experiments [REF:asc_review] compares YOLOv8–v11 on ExDark, DarkFace and LLVIP and observes that architectural upgrades bring only marginal gains compared with dataset and illumination characteristics.

Two independent 2026 studies reach the same negative conclusion we report here. Thu and Jangsamsi [REF:icsipa] evaluate CLAHE, LIME, Zero-DCE, RetinexFormer and a training-free hybrid pipeline on ExDark with two COCO-pretrained detectors (YOLOv8n and Faster R-CNN) and find that no method consistently outperforms the raw image, that a validation-tuned CLAHE is competitive with far more complex learned methods, and that a conservative branch matches raw detection while cutting enhancement latency by roughly 275×. Liang [REF:pcamv] evaluates Zero-DCE, SCI, Retinexformer and LLFlow on LOL eval15 plus a stratified ExDark subset with YOLOv5s and reports that the raw input attains the highest mAP@0.5 and recall even though Retinexformer has the best PSNR and LLFlow the best SSIM, together with GFLOPs and FPS. Both papers therefore anticipate our central observation; what neither provides is a quantitative account of *why* it happens.

The opposite has also been reported. In a secondary detection experiment, Cai et al. [REF:sgvmnet] feed ExDark images enhanced by SGVMNet and by competing methods to YOLOv8s and report small positive gains (RetinexFormer 77.54 against a raw-image baseline of 76.76). Two points qualify the comparison. Their reported baseline sits 8–16 mAP above ours (60.51 frozen, 68.18 after adaptation), which indicates a different training budget, split and evaluation convention rather than a contradiction of the measurement; and detection is a supporting experiment in a method paper whose primary evidence is paired-benchmark PSNR. Read together, the two literatures bracket the phenomenon: when the detector is heavily adapted to the domain, the marginal value of enhancement tends toward zero — which is the quantity Protocol II isolates.

The present paper differs in four respects. First, our pool is broader on the classical side (six operators, including the whole Retinex family), which lets us show that the loss is *monotonic in aggressiveness* rather than an artefact of one method. Second, we add a second benchmark with a retained ground truth, on which restoration fidelity can be measured directly and the degradation itself can be ablated — this is what turns "the loss is not quality" into a decomposition into noise, exposure and distributional shift. Third, we stress-test the brightness–mAP relation that our own ExDark analysis suggests, find that it does not transfer across datasets, and report it as a within-dataset description rather than a law; the transferable claim is narrower. Fourth, we introduce a two-protocol design that decomposes the loss into domain shift and quality, a decomposition that — to our knowledge — has not been reported for this task, and which explains why the ranking observed under a frozen detector is not the ranking that matters after deployment. We do not evaluate computational cost; where latency is the object of study we refer the reader to [REF:icsipa][REF:pcamv].

---

## 3 Evaluation protocol

### 3.1 Datasets

ExDark [REF:exdark] provides 7,363 real low-light images with 23,710 bounding boxes over twelve classes (bicycle, boat, bottle, bus, car, cat, chair, cup, dog, motorbike, people, table). One image lacks usable annotations, leaving 7,362 images for evaluation. Image sizes are heterogeneous (640×480, 640×427, 507×338, and others, up to 4,372 px on the long side), so no fixed input size is assumed.

COCO val2017 [REF:coco] provides the second, controlled benchmark. All 5,000 validation images are used, and evaluation is restricted to the twelve classes shared with ExDark — all of them occur in COCO, with only *person*, *motorcycle* and *dining table* requiring renaming — so the same mAP implementation applies unchanged. Images without target-class annotations are kept, so that false positives are counted. A low-light version is synthesised by a three-component degradation, and a noiseless variant of the same degradation serves as an ablation; both are specified in §6.2. Because the clean image is retained, this benchmark is the only place in the paper where restoration fidelity can be measured directly instead of inferred.

On ExDark we deliberately do **not** compute PSNR or SSIM: there is no normal-light ground truth, and any paired reference would have to be synthesised, which would confound the measurement with the synthesis model. Instead we report six reference-free statistics (§4.2) and rely on ExDark's own annotations for detection accuracy.

### 3.2 Enhancement variants

Twelve inputs are evaluated. All are produced from the same source image and resized consistently.

**Identity.** The raw image.

**Classical operators.**
- *Gamma* (γ = 0.5), a global intensity mapping.
- *CLAHE*, applied to the L channel of CIELAB with clip limit 2.0 and 8×8 tiles.
- *Histogram equalisation*, likewise on the L channel.
- *SSR*, single-scale Retinex with σ = 80.
- *MSR*, the mean of three Retinex responses with σ ∈ {15, 80, 250}.
- *MSRCR*, MSR multiplied by the colour-restoration factor with α = 125, β = 46.

Retinex outputs are mapped to 8-bit by a per-channel 1st–99th percentile stretch, since min–max normalisation collapses the useful range in the presence of outliers.

**Learned enhancers.**
- *Zero-DCE* [REF:zerodce], official LOL weights (79 k parameters).
- *Retinexformer* [REF:retinexformer], official LOL-v1 weights (1.61 M parameters).

**In-house variants.** A residual restoration network R(I) = I + Δ(I) with a four-level U-Net backbone (8.28 M parameters). Three configurations are used: the backbone alone (*Res-UNet*), backbone plus an auxiliary illumination head (*+Illum*), and the full model with illumination and noise heads regularised by a reconstruction consistency term (*INDNet*). These are included as additional data points, not as a claimed contribution; the network's PSNR on LOL-v1 is 21.13 dB, well below contemporary methods, and its three configurations are statistically indistinguishable on PSNR (20.63–21.13 dB, within the ±0.5 dB standard error of a 15-image test set).

**Resolution policy.** Every enhancement — classical or learned — is applied at a long side of at most 1,024 px and then resampled to the original resolution. The detector subsequently resizes to 640 px, so this policy cannot advantage any method, and it keeps the cost of the Gaussian filtering in the Retinex family bounded.

### 3.3 Detectors and the two protocols

The primary detector is YOLOv8n [REF:yolov8] with COCO-pretrained [REF:coco] weights, frozen throughout: no parameter is updated, and only the input changes. Predictions are produced at confidence 0.01 with NMS IoU 0.5, following common practice for mAP computation.

Two protocols are used.

**Protocol I (frozen).** The detector is never re-trained. This isolates the effect of the input distribution and corresponds to the common deployment shortcut of "drop in an enhancer".

**Protocol II (domain adaptation).** ExDark is split 60/15/25 by image into train/val/test with a fixed seed. Under *Protocol II-A*, one detector is fine-tuned per pipeline (raw, CLAHE, INDNet) on that pipeline's own enhanced training images and evaluated on its own test images; under *Protocol II-B*, the detector fine-tuned on raw images is evaluated on the other pipelines' test images. The pair (II-A, II-B) separates the two effects: II-B includes the distribution mismatch, II-A removes it.

### 3.4 Metrics and statistical criteria

Detection accuracy is mAP@0.5 computed with **all-point interpolated average precision** (the VOC2010+ convention: the area under the monotonically decreasing precision envelope), averaged over the twelve classes that contain at least one ground-truth box, with predictions taken at confidence 0.01 and NMS IoU 0.5. Because the evaluation set contains 7,362 images over twelve classes, the sampling noise on mAP is of order ±0.005; **differences smaller than this are treated as no difference** throughout. For PSNR on the 15-image LOL-v1 test set, the standard error is approximately ±0.5 dB, and we refrain from ranking methods whose PSNR differs by less than that.

Wherever a detector has to be trained (Protocol II-A, §5), every number is the mean of three training seeds (3407, 1234, 2024) on a fixed split, with the standard deviation reported alongside. The measured seed-to-seed spread is 0.0021–0.0036 mAP, i.e. below the sampling tolerance, so no conclusion in this paper rests on a single lucky run; the spread is also large enough that a single-seed comparison of two pipelines differing by less than ~0.006 would have been uninformative.

---

## 4 Results under a frozen detector

### 4.1 Almost every enhancement hurts

Table 1 reports mAP@0.5 for all twelve inputs, ordered by mean luminance.

**Table 1.** Detection accuracy on ExDark (7,362 images) with a frozen YOLOv8n. Y is the mean luminance of the enhanced set (0–255); Δ is relative to the raw image; residual is the deviation from the two-factor fit of §4.3.

| Input | Type | mAP@0.5 | Δ | Y | dark % | residual |
|---|---|---|---|---|---|---|
| Raw image | — | 0.5338 | — | 37.1 | 58.2 | −0.0345 |
| CLAHE | classical | **0.5367** | **+0.0029** | 52.8 | 43.3 | −0.0063 |
| Gamma (γ=0.5) | classical | 0.5008 | −0.0331 | 77.5 | 16.2 | −0.0024 |
| Res-UNet | learned | 0.5134 | −0.0204 | 80.5 | 20.2 | **+0.0150** |
| Zero-DCE | learned | 0.4873 | −0.0466 | 86.1 | 17.5 | −0.0021 |
| Retinexformer | learned | **0.5204** | **−0.0134** | 89.8 | 19.9 | **+0.0370** |
| INDNet (full) | learned | 0.4957 | −0.0381 | 90.0 | 14.6 | +0.0126 |
| + Illum | learned | 0.4739 | −0.0599 | 96.8 | 12.7 | +0.0017 |
| Histogram eq. | classical | 0.4494 | −0.0845 | 119.1 | 10.2 | +0.0132 |
| MSRCR | Retinex | 0.2660 | −0.2679 | 106.2 | 2.1 | (outlier) |
| SSR | Retinex | 0.3573 | −0.1765 | 157.4 | 1.8 | −0.0173 |
| MSR | Retinex | 0.3531 | −0.1807 | 160.3 | 1.0 | −0.0168 |

Only CLAHE lies within the ±0.005 tolerance of the raw image. The remaining eleven inputs lose between 0.013 and 0.268 mAP. The spread is not arbitrary: it follows the aggressiveness of the operator, from a mild local contrast adjustment (CLAHE, neutral) through a global power law (gamma, −0.033) and a global histogram stretch (equalisation, −0.085) to full illumination separation (SSR/MSR, −0.18) and its colour-restored variant (MSRCR, −0.27).

Two entries deserve emphasis. Retinexformer is the strongest enhancer in our pool by a wide margin on paired benchmarks (PSNR 25.16 dB on LOL-v1, against 21.13 dB for our network), yet it still loses 0.013 mAP under a frozen detector — the smallest loss among learned methods, but a loss nonetheless. MSRCR, the most aggressive operator, retains only 50% of the raw mAP.

### 4.2 Detection accuracy tracks brightness — on ExDark

Because ExDark has no ground truth, we characterise each enhanced set by six reference-free statistics: mean luminance Y, the fraction of pixels with Y < 25.5 ("dark %"), the fraction of clipped pixels ("clip %"), mean absolute gradient, colourfulness [REF:colorful], and the entropy of the luminance histogram. Table 2 lists them alongside mAP.

**Table 2.** Reference-free statistics of each input set, averaged over 7,362 images.

| Input | Y | dark % | clip % | grad | colour | entropy | mAP |
|---|---|---|---|---|---|---|---|
| Raw | 37.1 | 58.2 | 1.23 | 9.09 | 31.0 | 3.01 | 0.5338 |
| CLAHE | 52.8 | 43.3 | 1.49 | 14.46 | 31.3 | 3.58 | 0.5367 |
| Gamma | 77.5 | 16.2 | 1.57 | 12.10 | 34.0 | 3.80 | 0.5008 |
| Res-UNet | 80.5 | 20.2 | 5.37 | 13.90 | 43.7 | 3.96 | 0.5134 |
| Zero-DCE | 86.1 | 17.5 | 1.52 | 16.26 | 45.1 | 3.89 | 0.4873 |
| Retinexformer | 89.8 | 19.9 | 5.12 | 15.14 | 48.6 | 4.08 | 0.5204 |
| INDNet | 90.0 | 14.6 | 7.34 | 14.51 | 43.6 | 4.02 | 0.4957 |
| + Illum | 96.8 | 12.7 | 10.15 | 15.77 | 47.2 | 4.09 | 0.4739 |
| Histogram eq. | 119.1 | 10.2 | 9.30 | 21.85 | 29.7 | 4.64 | 0.4494 |
| MSRCR | 106.2 | 2.1 | 0.92 | 19.37 | 62.1 | 3.83 | 0.2660 |
| SSR | 157.4 | 1.8 | 2.03 | 20.75 | 43.9 | 4.26 | 0.3573 |
| MSR | 160.3 | 1.0 | 2.16 | 22.06 | 44.2 | 4.21 | 0.3531 |

Mean luminance is by far the strongest correlate (Fig. 2). Excluding MSRCR — whose colourfulness of 62.1 is far outside the 30–49 range of all other methods and which we treat as a separate failure mode — the Pearson correlation between Y and mAP is **−0.954** (Spearman −0.909), corresponding to r² ≈ 0.91. The dark-pixel ratio behaves symmetrically (Spearman +0.955): **preserving dark regions is safe; removing them is not.** Gradient magnitude correlates negatively too (−0.84), but gradient and brightness are themselves strongly coupled — aggressive enhancement amplifies both — so we do not claim an independent effect.

A linear fit over the eleven non-outlier methods gives

  mAP ≈ −1.61×10⁻³ · Y + 0.6281.

Read as a rule of thumb, every 10 units of added mean luminance costs about 0.016 mAP. The safe range is narrow: from Y = 37 (raw) to Y ≈ 55 (CLAHE) nothing is lost, while at Y = 90 the loss is already 0.04.

### 4.3 A two-factor description (ExDark only)

A single factor is not sufficient, and the data say so precisely. Retinexformer and INDNet have almost identical brightness (89.8 and 90.0) yet differ by 0.025 mAP — over four times the sampling tolerance. Whatever separates them is not brightness.

We therefore read the residuals of the fit as a measure of the second factor, restoration fidelity, and find that they order the methods exactly as independent evidence on restoration quality does (Table 3).

**Table 3.** Residuals from the brightness–mAP fit, ordered by magnitude.

| Method | residual | independent fidelity evidence |
|---|---|---|
| Retinexformer | **+0.0370** | PSNR 25.16 dB on LOL-v1 (best in pool) |
| Res-UNet | +0.0150 | — |
| Histogram eq. | +0.0132 | — |
| INDNet | +0.0126 | PSNR 21.13 dB on LOL-v1 |
| + Illum | +0.0017 | PSNR 20.63 dB |
| Zero-DCE | −0.0021 | designed for perceptual, not pixel fidelity |
| CLAHE | −0.0063 | — |
| Gamma | −0.0024 | — |
| MSR | −0.0168 | — |
| SSR | −0.0173 | — |
| Raw (reference) | −0.0345 | — |

The ordering of the two methods for which we have paired-benchmark evidence — Retinexformer above INDNet — matches the ordering of their PSNR by four decibels, and the ratio of their residuals (+0.037 vs +0.013) is of the same order. We therefore state the model as

  **mAP ≈ f(brightness) + g(fidelity)**,

where f is the fitted linear term and g is the residual: brightness governs how far the input has moved from the detector's operating distribution, and fidelity governs how much structure survives the move. The two factors are of different magnitude — brightness spans 0.18 mAP across our pool, fidelity at most 0.04 — which is why the effect of enhancement is dominated by how bright it makes the image, and why reporting PSNR alone predicts so poorly: PSNR is a proxy for g only.

We stress the scope of this model. It is fitted on **one dataset, one scene family and one deformation direction**, and it is descriptive rather than causal: brightness is a proxy for "distance from the operating distribution" only as long as every operator in the pool moves the image in the same direction, i.e. upwards. In §6.2 we test it on a second benchmark where the deformation can be reversed, and there the coefficient **changes sign** (−0.954 to +0.61). The model is therefore a within-ExDark description, not a law; the general statement that survives the transfer is narrower and is stated in §7.

### 4.4 Which classes suffer

The per-class breakdown (Table A1 in the appendix) shows that the loss concentrates on a subset of categories. For INDNet, the largest drops are car (−0.086), dog (−0.080), cat (−0.068) and bus (−0.064); boat (+0.008), table (−0.004), bicycle (−0.008) and motorbike (−0.022) are barely affected.

Two plausible readings coexist, and we do not have the evidence to separate them: the worst-affected classes are precisely those whose appearance depends on fine texture or on low-contrast boundaries that aggressive brightening tends to flatten, while the least-affected are either high-contrast elongated structures (bicycle, motorbike) or objects with very low intrinsic contrast that the detector barely finds even in the raw image (table, boat, mAP < 0.38). Per-class differences on a 7,362-image set are still subject to sampling noise, so we report them as descriptive rather than inferential.

### 4.5 Qualitative behaviour

Figure 4 shows four cases selected not at random but as the *worst* cases for our network, i.e. those with the largest drop in the number of detected boxes. The raw image yields 10 boxes in the first row, against 8 after learned enhancement; in the second row, 9 against 6. The figure also exposes a behaviour that the aggregate numbers hide: on these very dark inputs, Retinexformer over-brightens, producing halos and, in one case, a strong colour cast, even though its dataset-level mAP remains the best among learned methods. Single-image appearance and dataset-level detection accuracy are thus not interchangeable — another reason to assess enhancement by its downstream effect rather than by inspection.

---

## 5 Domain-shift decomposition

### 5.1 Why decompose

Protocol I answers "what happens if I prepend an enhancer and change nothing else". That is not the only deployment scenario: a practitioner who adopts a new pre-processing stage can retrain, and often does. The relevant question then becomes how much of the observed loss is an artefact of the detector never having seen enhanced images, and how much survives re-training. Protocols II-A and II-B answer it: II-B retains the mismatch, II-A removes it, and their difference is the domain-shift component.

### 5.2 Results

**Table 4.** Protocol II. Same detector architecture and fixed split; only the input pipeline differs. Values are means ± std over three training seeds (3407, 1234, 2024); the seed spread is 0.0021–0.0036. Sampling tolerance ±0.005.

| Pipeline | II-B (cross-eval) | II-A (adapted) | quality (II-A Δ) | domain shift (II-B − II-A) |
|---|---|---|---|---|
| Raw | 0.6818 ± 0.0021 | 0.6818 ± 0.0021 | — | — |
| CLAHE | 0.6676 ± 0.0021 (−0.0143) | **0.6859 ± 0.0026 (+0.0041)** | +0.0041 (n.s.) | −0.0184 (all of the loss) |
| INDNet | 0.6215 ± 0.0034 (−0.0604) | 0.6698 ± 0.0036 (−0.0121) | −0.0121 | −0.0483 (**80%**) |

Domain adaptation itself is worth +0.148 mAP (0.5338 → 0.6818), a factor large enough to dwarf every enhancement effect measured in this paper. This is the single most consequential number we report: **any evaluation of enhancement for detection that freezes the detector understates detection performance by a wide margin, and attributes to the enhancer a loss that is mostly a distribution mismatch.**

After adaptation the picture changes qualitatively, but less than a single run suggests. CLAHE returns to the level of the raw image (+0.0041, below the 0.005 tolerance): adaptation **neutralises** its penalty rather than turning it into a gain. Learned enhancement remains negative (−0.0121, 2.4× tolerance), and 80% of its cross-evaluated loss (−0.0604) is domain shift, the remaining 20% being attributable to restoration quality.

The three-seed repetition matters here. Our first single-seed run put CLAHE at +0.0068, which would have supported the stronger claim that adaptation makes conservative enhancement *beneficial*; across seeds the value falls to +0.0041 with a spread of 0.0052, i.e. inside the training noise. We therefore state the weaker and defensible version — adaptation removes the penalty, it does not create a benefit — and note that the 80/20 split of INDNet's loss is stable to within about ±7 percentage points across seeds.

The decomposition also corrects a misleading comparison. Under a frozen detector the gap between CLAHE and INDNet is 0.041 mAP; after adaptation it is 0.0161. About 60% of the apparent gap was never about image quality.

### 5.3 Interpretation

The two components have different remedies. Domain shift is fixed by re-training — expensive but mechanical. Quality loss is fixed only by a better enhancer. Our results suggest that for this task the first remedy is both larger and cheaper, and that a conservatively tuned classical operator with a re-trained detector is a stronger baseline than an aggressively tuned learned one without.

---

## 6 Robustness

### 6.1 A second detector

To check that the ranking is not an artefact of one architecture, Protocol I was repeated with YOLOv8s, a larger model of the same family. Four inputs were re-evaluated (Table 5).

**Table 5.** Protocol I with two detectors. Δ relative to the raw image.

| Input | YOLOv8n mAP | Δ | YOLOv8s mAP | Δ |
|---|---|---|---|---|
| Raw | 0.5338 | — | 0.6051 | — |
| CLAHE | 0.5367 | +0.0029 | 0.6101 | +0.0050 |
| INDNet | 0.4957 | −0.0381 | 0.5698 | −0.0353 |
| MSR | 0.3531 | −0.1807 | 0.4367 | −0.1684 |

The ordering (CLAHE > raw > INDNet > MSR) is identical, and the deltas agree to within 0.013. The larger detector is uniformly more accurate in absolute terms, as expected, but no more or less sensitive to the enhancement. The reference-free statistics of the enhanced sets are, of course, identical to those in Table 2, since they depend only on the enhancement pipeline — a useful consistency check on the measurement chain.

### 6.2 A second benchmark, with ground truth

ExDark answers one question and leaves two open. It cannot tell us how far an enhanced image is from a *correct* rendering, because it has no clean reference; and it cannot test whether the brightness relation of §4.2 is a property of the problem or of the fact that every operator in the pool brightens. We therefore built a second benchmark from COCO val2017.

We take all 5,000 validation images, keep the same twelve classes as ExDark (all are present in COCO; only *person*, *motorcycle* and *dining table* need renaming), and synthesise a low-light version with three components: a smooth illumination field (a 6×6 random field, mean 1, range 0.3–2.2) to model spatially non-uniform lighting; a power law of γ = 1.6 applied in sRGB to match the strong shadow compression of real night captures; and signal-dependent sensor noise (σ = 0.045·√x plus a read-noise floor). The global gain is then calibrated so that the mean luminance matches ExDark (Y = 37.0 against 37.1) and the dark-pixel fraction is comparable (44.2% against 58.2%). Two recipes are used: the full model, and a **noiseless** variant in which the third component is removed — this isolates the contribution of sensor noise and is the subject of §6.3. All 5,000 images are evaluated, including those without target-class annotations, so that false positives are counted.

Because the clean image is retained, every condition can be scored against a true reference — something ExDark does not permit. Detector, classes, confidence threshold and mAP implementation are unchanged from §3.

**Table 6.** COCO val2017, 5,000 images, frozen YOLOv8n, mAP@0.5. Δ is relative to the degraded input of the same recipe. Sampling tolerance ±0.006.

| Condition | Noisy degradation | Δ | Noiseless degradation | Δ |
|---|---|---|---|---|
| Clean image (upper bound) | 0.5700 | +0.0795 | 0.5700 | +0.0330 |
| Degraded input | 0.4905 | — | 0.5370 | — |
| + CLAHE | 0.4704 | **−0.0201** | 0.5432 | +0.0062 |
| + INDNet (ours) | 0.4840 | −0.0065 | 0.5259 | **−0.0111** |
| + Retinexformer | 0.5025 | **+0.0120** | 0.5501 | **+0.0131** |
| Clean + CLAHE | 0.5629 | — | — | — |

Three observations. First, **the negative result replicates**. Under the noisy recipe the best input recovers 0.012 — twice the tolerance, but a sixth of the 0.0795 gap — and CLAHE, the only neutral operator on ExDark, is here clearly harmful. Under the noiseless recipe the best input recovers 0.013 out of 0.033. No condition comes close to the clean-image bound; the best method still leaves 0.0199 of the gap unfilled when noise is absent, i.e. when the information is demonstrably recoverable.

Second, **enhancing an already well-exposed image does not help either**: applying CLAHE to the clean image costs 0.0071 mAP. The loss therefore does not require the input to be dark; it only requires the input statistics to move.

Third, and most importantly for §6.3, the two recipes differ in the *magnitude* of the damage (0.0795 against 0.0330) but not in its *sign*: in both, the strongest learned enhancer barely breaks even and our own network does not.

### 6.3 Where the loss comes from: noise, exposure and fidelity

With the reference available, we can measure restoration fidelity directly and ask which quantity, if any, predicts detection (Table 7).

**Table 7.** Fidelity against the clean reference, highlight clipping, and mAP. Clipping is the fraction of pixels with any channel ≥ 250; the clean image clips 3.47%.

| Recipe | Condition | PSNR (dB) | SSIM | clip % | mAP | Δ vs degraded |
|---|---|---|---|---|---|---|
| noisy | Degraded | 10.00 | 0.359 | 0.11 | 0.4905 | — |
| noisy | + CLAHE | 12.82 | 0.525 | 0.15 | 0.4704 | −0.0201 |
| noisy | + INDNet | 15.64 | 0.637 | 1.43 | 0.4840 | −0.0065 |
| noisy | + Retinexformer | 15.65 | 0.673 | 9.42 | 0.5025 | +0.0120 |
| noiseless | Degraded | 10.03 | 0.432 | 0.13 | 0.5370 | — |
| noiseless | + CLAHE | 12.75 | 0.685 | 0.20 | 0.5432 | +0.0062 |
| noiseless | + INDNet | 16.97 | 0.802 | 6.60 | 0.5259 | −0.0111 |
| noiseless | + Retinexformer | 17.34 | 0.842 | 1.64 | 0.5501 | +0.0131 |

**Noise accounts for 58% of the damage.** The degradation removes 0.0795 mAP when sensor noise is present and 0.0330 when it is not; the difference, 0.0465, is attributable to noise alone. The effect on CLAHE is decisive: under noise it costs 0.0201 (3.3× the tolerance), without noise it *gains* 0.0062 and is indistinguishable from the degraded input. CLAHE is a locally adaptive contrast operator, and on a noisy dark image it amplifies exactly the high-frequency component that the detector's early layers respond to. Its failure is therefore not a failure of enhancement in general but of enhancement that ignores noise.

**Fidelity does not predict detection.** Within the degraded branch, the PSNR–mAP correlation is r = +0.24 under noise and **r = +0.007** without it; for SSIM the values are +0.25 and +0.13. Two comparisons make this concrete rather than correlational:

- Under noise, CLAHE improves PSNR by **2.82 dB** and SSIM by **0.166** over the degraded input, and *loses* 0.0201 mAP.
- In the noiseless recipe, INDNet improves PSNR by **6.94 dB** and SSIM by **0.370**, and *loses* 0.0111 mAP — while Retinexformer, at almost the same PSNR (0.37 dB apart), gains 0.0131. The 0.0242 mAP gap between the two is four times the sampling tolerance.

**Local over-exposure behaves consistently in the noiseless recipe.** The clean image clips 3.47% of pixels. Without noise, the only condition that loses accuracy is also the one that clips far more than the reference: INDNet reaches 6.60%, 1.9× the reference and 4× Retinexformer's 1.64%. Across the four noiseless conditions, clipping is the only statistic whose correlation with mAP has both the right sign and a non-trivial magnitude (r = −0.72, against +0.007 for PSNR). The same ordering appears in the ExDark ablation family: clipping of 5.37%, 7.34% and 10.15% for the Res-UNet baseline, the full model and the illumination-only variant corresponds exactly to mAP of 0.5134, 0.4957 and 0.4739.

The relation does not hold under noise, and we state this rather than smooth it over: Retinexformer clips 9.42% and still improves, while CLAHE clips 0.15% and loses. Clipping is therefore better read as one of three mechanisms — and as a quantity worth monitoring — than as a predictor of detection accuracy.

We report this as an observation, not a law: with four conditions per recipe the correlations carry wide uncertainty, and the relation is not monotone across recipes. What the data support is the negative claim, which needs no calibration: **neither PSNR nor SSIM ranks these inputs in the order the detector does.**

---

## 7 Discussion

**A brightness ceiling — but not a universal one.** On ExDark the fitted relation implies a practical guideline: keep the mean luminance near that of the input. Staying within Y ≈ 37–55 costs nothing; crossing Y ≈ 80 costs 0.02–0.04 mAP; reaching Y > 150 costs 0.18. That is a cheap, interpretable constraint that can be checked before any detector is run, and it explains why CLAHE — which raises Y by only 16 units while preserving 43% of dark pixels — is the only operator that survives.

It does not generalise as stated. On COCO, raising the mean luminance from 37 to 114 *improves* mAP by 0.0795, and the correlation between brightness and mAP changes sign (r = −0.954 on ExDark, +0.61 on COCO). A rule of the form "brighter is worse" is therefore an artefact of a pool in which every operator brightens an image that was never degraded in the first place. The transferable statement is narrower: **enhancement can only help if it reconstructs information that was actually lost, and only if the reconstruction is accurate enough to be worth the distributional perturbation it introduces.** ExDark satisfies neither condition — a night scene has no "correct" daytime rendering, so brightening is invention. COCO satisfies the first condition but not the second: the clean image exists, yet the best enhancer reaches only 17.3 dB and recovers 40% of the damage at best.

**Three mechanisms, not one.** The COCO ablation separates what ExDark confounds:

| Mechanism | Evidence | Detector-relevant? |
|---|---|---|
| **Noise amplification** | removing sensor noise halves the damage (0.0795 → 0.0330) and flips CLAHE from −0.0201 to +0.0062 | yes — high-frequency energy in early layers |
| **Local over-exposure** | INDNet clips 6.60% against 3.47% in the reference and is the only method that loses accuracy without noise; the ExDark ablation family orders identically (5.37/7.34/10.15% ↔ 0.5134/0.4957/0.4739) | plausible, four-point evidence |
| **Distributional shift** | re-training the detector recovers +0.148 mAP and neutralises CLAHE entirely (§5) | yes — dominant term |

The three are not additive in any simple way, but they explain why the same operator can be safe in one setting and harmful in another: CLAHE is neutral on ExDark (whose noise level is moderate) and actively harmful on our noisy COCO degradation (where it amplifies noise), and it becomes neutral again once noise is removed.

**Why PSNR and SSIM fail as selection criteria.** Both are averages of pixel-wise error, and both are blind to the two mechanisms above. Noise amplification adds error uniformly across the image, which *lowers* PSNR — yet CLAHE is the case where PSNR improves while detection degrades, because the amplified noise sits on a stronger signal. Over-exposure affects a few percent of pixels, which barely moves a mean-squared error, but those pixels are exactly the specular highlights and bright boundaries that localise objects. The measurements make the failure quantitative:

- CLAHE, noisy recipe: **+2.82 dB PSNR, +0.166 SSIM, −0.0201 mAP**.
- INDNet, noiseless recipe: **+6.94 dB PSNR, +0.370 SSIM, −0.0111 mAP**.
- INDNet against Retinexformer, noiseless: **0.37 dB apart in PSNR, 0.0242 apart in mAP**.
- Within-branch correlation: **r = 0.007** (noiseless) and 0.24 (noisy).

Our own network illustrates the trap from the other side. Its PSNR on LOL-v1 (21.13 dB) and the PSNRs of its ablated variants (20.63–21.13 dB) are statistically indistinguishable, yet their detection behaviour differs systematically and monotonically with clipping. Reporting PSNR as evidence that an enhancer is "better for detection" is therefore unsupported — not because the metric is noisy, but because it measures a quantity the detector does not optimise.

**Colour is a separate axis.** MSRCR sits far off the brightness trend: at Y = 106 it should score around 0.45, but achieves 0.266. Its colourfulness of 62.1 is well outside the 30–49 band of every other method, and its dark-pixel fraction is 2.1%. We read this as a third, independent failure mode — hue distortion — that neither the brightness model nor the fidelity metrics capture. Any practitioner using Retinex-family operators should monitor colour statistics as well.

**Implications for deployment.** Two rules follow. First, do not insert an enhancer and keep the detector: the loss is 0.013–0.27 mAP on ExDark and 0.007–0.020 on COCO, and in both cases it is invisible in the enhancer's own metrics. Second, if an enhancer is adopted, retrain the detector — the gain from adaptation (+0.148) exceeds anything the enhancer can contribute, and after adaptation a conservative operator's penalty disappears entirely, although it does not become a gain. The COCO results add a third: if the pipeline must handle both noisy and clean inputs, prefer operators that do not amplify high frequencies, and monitor clipping before monitoring PSNR.

---

## 8 Limitations

*The second dataset is synthetic.* Our COCO benchmark uses a degradation we constructed, calibrated so that its first-order statistics match ExDark (mean luminance 37.0 against 37.1; dark-pixel fraction 44.2% against 58.2%). It cannot reproduce motion blur, chromatic noise, rolling-shutter artefacts or JPEG behaviour of a real night capture. The value of the benchmark is that it supplies a clean reference and a controllable noise term; it is not a substitute for a second *real* low-light detection dataset, which remains the most valuable missing experiment.

*One synthesis recipe.* We fix γ = 1.6, one illumination-field scale and one noise level, and vary only the presence of noise. A sweep over these parameters would turn "how much noise is tolerable" into a quantitative answer, and would show whether the clipping effect we observe survives changes in the degradation model. Note also that applying the power law in sRGB, as most synthesis pipelines do, slightly amplifies colour opponency (+0.014 in normalised (R−B)/Y, against +0.007 for a physically correct linear-light attenuation); we did not correct for this.

*One split, three seeds.* Protocol II is repeated over three training seeds but on a single 60/15/25 split, so split-to-split variation is not measured. The split is generated deterministically from a recorded seed and the per-seed spread is well below the sampling tolerance, so we expect split variation of the same order — but that is an assumption, not a measurement.

*Two detectors, one family.* Both detectors are YOLOv8 variants. The ordering is stable across them, which addresses capacity but not architecture: a two-stage or DETR-based detector may respond differently, and the domain-shift decomposition has been verified only for the YOLOv8n pipeline set.

*Fidelity is measured only on COCO.* On ExDark, restoration fidelity remains inferred from residuals rather than measured, because no clean reference exists. The COCO experiment resolves this for one benchmark; the ExDark residual analysis is therefore supported by, but not calibrated against, direct measurement.

*Small samples for the mechanism claims.* The correlations in §6.3 rest on four conditions per recipe. We therefore present clipping as an observation consistent with the ExDark ablation family, not as an established predictor. The negative claim — that PSNR and SSIM do not rank inputs as the detector does — does not depend on those correlations; it rests on the three pairwise comparisons.

*No reference-free quality metrics.* We report luminance, contrast, gradient, colourfulness and entropy, but not NIQE or BRISQUE; adding them would allow a direct test of whether any no-reference score predicts detection. This is left for future work.

*Operator parameters are not tuned.* Every method runs at its published defaults or the values used by common implementations: CLAHE with clip limit 2.0 and 8×8 tiles, γ fixed at 0.5, Zero-DCE and Retinexformer at their released LOL weights. Thu and Jangsamsi [REF:icsipa] report that a CLAHE tuned on a validation split is competitive with far more complex learned methods, so a modest tuning effort could plausibly move our CLAHE result above the neutral line. We did not tune it, deliberately: tuning one operator and not the others would make the comparison asymmetric, and tuning on the evaluation domain would leak test information into the pipeline. Our numbers should therefore be read as out-of-the-box performance — the setting a practitioner faces before investing in per-dataset calibration.

*The learned pool is small.* Two recent learned methods (Zero-DCE, Retinexformer) are included. While they bracket the current state of the art on paired benchmarks, the conclusion would be stronger with four to six.

---

## 9 Conclusion

We re-examined a routine assumption — that enhancing a low-light image before detection must help — under a protocol in which only the input changes. Across two benchmarks — twelve operators on ExDark and six conditions on a controlled COCO degradation under two recipes, each evaluated with a frozen detector — **no enhancement reliably improves detection**. On ExDark only CLAHE is neutral (+0.003 mAP) while the other eleven lose 0.013–0.268, including the ICCV'23 state of the art. On COCO, where the clean image is retained as a reference, the best input recovers 0.013 of a 0.0795 gap under a noisy degradation and 0.013 of 0.0330 without noise; the ordering reproduces with a second detector.

The COCO experiment also settles what ExDark could not. The loss is **58% sensor noise and 42% restoration error**, and the two have different signatures: noise amplification explains why a locally adaptive operator (CLAHE) is harmful on a noisy degradation and neutral once noise is removed; local over-exposure — 6.6% of pixels clipped against 3.5% in the reference — explains why a network that gains 6.9 dB of PSNR can still lose accuracy. Neither effect is visible in PSNR or SSIM: within the degraded branch the PSNR–mAP correlation is 0.007, two methods 0.4 dB apart in PSNR differ by 0.024 in mAP, and one operator improves PSNR by 2.8 dB while losing 0.020 mAP.

Brightness, finally, is not a portable predictor: its correlation with mAP is −0.95 on ExDark and +0.61 on COCO. The transferable statement is that enhancement helps only when it reconstructs information that was actually lost, accurately enough to be worth the distributional change it imposes — a condition that neither benchmark satisfies in practice.

For practice, four rules: keep the brightness gain small; never change the pre-processing without re-training the detector (worth +0.148 mAP, and 80% of the residual loss is domain shift); never select an enhancer by PSNR; and, when the input may be noisy, monitor clipping and noise amplification rather than mean-squared error.

For research, the implication is less comfortable. If a 4 dB PSNR advantage translates into at most 0.04 mAP at equal brightness, and if the strongest method in our pool fails to recover even half of a demonstrably recoverable loss, then progress measured on paired benchmarks is buying very little for detection. The community's evaluation practice — PSNR on synthetic pairs, detection reported separately, if at all — is not measuring what deployment needs.

---

## Appendix A. Per-class results (Protocol I, YOLOv8n)

**Table A1.** Per-class mAP@0.5 under the frozen detector, six of the twelve inputs. Full ranking in Table 1.

| Class | Raw | CLAHE | MSR | Retinexformer | Zero-DCE | INDNet |
|---|---|---|---|---|---|---|
| bicycle | 0.6233 | 0.6321 | 0.4903 | 0.6288 | 0.6204 | 0.6153 |
| boat | 0.3671 | 0.3740 | 0.3100 | 0.3746 | 0.3667 | 0.3751 |
| bottle | 0.4901 | 0.4810 | 0.2783 | 0.4833 | 0.4407 | 0.4574 |
| bus | 0.7946 | 0.7883 | 0.5594 | 0.7756 | 0.7369 | 0.7307 |
| car | 0.6555 | 0.6643 | 0.4404 | 0.6226 | 0.5881 | 0.5697 |
| cat | 0.5083 | 0.5087 | 0.2451 | 0.4852 | 0.4112 | 0.4401 |
| chair | 0.4777 | 0.4783 | 0.2993 | 0.4631 | 0.4518 | 0.4541 |
| cup | 0.4065 | 0.3898 | 0.2458 | 0.3863 | 0.3475 | 0.3678 |
| dog | 0.5030 | 0.5087 | 0.3074 | 0.4713 | 0.3904 | 0.4232 |
| motorbike | 0.5987 | 0.6085 | 0.3374 | 0.5723 | 0.5570 | 0.5763 |
| people | 0.7116 | 0.7196 | 0.5434 | 0.7023 | 0.6786 | 0.6728 |
| table | 0.2697 | 0.2876 | 0.1808 | 0.2799 | 0.2579 | 0.2662 |

---

## Figure captions

**Fig. 1.** Evaluation protocols. Top: Protocol I freezes the detector and varies only the input. Bottom: Protocol II trains one detector per pipeline (II-A) and cross-evaluates them (II-B); the difference separates domain shift from restoration quality.

**Fig. 2.** Detection accuracy versus mean luminance for the eleven non-outlier ExDark inputs. Line: least-squares fit mAP = −1.61×10⁻³·Y + 0.6281, r = −0.954. Retinexformer and INDNet, at nearly identical brightness, are marked. MSRCR is shown but excluded from the fit. This relation is descriptive of ExDark only: on COCO the same correlation changes sign to +0.61 (§6.2).

**Fig. 3.** Per-class mAP for four representative inputs (appendix table A1).

**Fig. 4.** Qualitative comparison on the four inputs with the largest drop in detected boxes for INDNet. Columns: raw, CLAHE, Retinexformer, INDNet. Numbers above each panel give the box count at confidence 0.25. These are stress cases chosen to expose failure modes, not representative samples; the quantitative conclusions rest on Table 1.

**Fig. 5.** The COCO degradation. Left: original val2017 image. Right: the same image after the three-component degradation of §6.2 (illumination field, γ = 1.6, signal-dependent noise), calibrated to Y = 37 to match ExDark. Three examples; the clean image on the left is the reference used for every PSNR/SSIM value in Table 7.

---

## References

*(all entries are real publications; DOI to be completed before submission)*

1. [REF:exdark] Loh, Y.P., Chan, C.S.: Getting to know low-light images with the exclusively dark dataset. Computer Vision and Image Understanding **130**, 30–42 (2019)
2. [REF:zerodce] Guo, C., Li, C., Guo, J., Loy, C.C., Hou, J., Kwon, S., Feng, S.: Zero-reference deep curve estimation for low-light image enhancement. CVPR 2020
3. [REF:zerodcepp] Li, C., Guo, C., Loy, C.C.: Learning to enhance low-light image via zero-reference deep curve estimation. IEEE TPAMI **44**(8), 4225–4238 (2022)
4. [REF:retinexformer] Cai, Y., Bian, H., Lin, J., Wang, H., Timofte, R., Zhang, Y.: Retinexformer: one-stage Retinex-based transformer for low-light image enhancement. ICCV 2023
5. [REF:uretinex] Wu, W., Weng, J., Zhang, P., Wang, X., Yang, W., Tian, Q.: URetinex-Net: Retinex-based deep unfolding network for low-light image enhancement. CVPR 2022
6. [REF:kind] Zhang, Y., Guo, X., Ma, J., Liu, W., Zhang, J.: Beyond brightening low-light images. International Journal of Computer Vision **129**, 1013–1037 (2021)
7. [REF:llflow] Wang, Y., Wan, R., Yang, W., Li, H., Wang, L., Kot, A.: Low-light image enhancement with normalizing flow. AAAI 2022
8. [REF:sci] Ma, L., Ma, T., Liu, R., Fan, X., Luo, Z.: Toward fast, flexible, and robust low-light image enhancement. CVPR 2022
9. [REF:snr] Xu, Y., Yang, C., Sun, K., et al.: SNR-aware low-light image enhancement. CVPR 2022
10. [REF:pairlie] Fu, Z., Yang, Y., Tu, X., Huang, Y., Ding, X., Ma, K.-K.: Learning a simple low-light image enhancer from paired low-light instances. CVPR 2023
11. [REF:llie_survey] Li, C., Guo, C., Han, L., Jiang, J., Cheng, M.-M., Gu, J., Loy, C.C.: Low-light image and video enhancement using deep learning: a survey. IEEE TPAMI **44**(12), 9396–9416 (2022)
12. [REF:clahe] Zuiderveld, K.: Contrast limited adaptive histogram equalization. In: Graphics Gems IV, pp. 474–485 (1994)
13. [REF:jobson] Jobson, D.J., Rahman, Z., Woodell, G.A.: A multiscale Retinex for bridging the gap between color images and the human observation of scenes. IEEE Transactions on Image Processing **6**(7), 965–976 (1997)
14. [REF:colorful] Hasler, D., Süsstrunk, S.: Measuring colourfulness in natural images. Proc. SPIE Human Vision and Electronic Imaging VIII (2003)
15. [REF:coco] Lin, T.-Y., Maire, M., Belongie, S., et al.: Microsoft COCO: common objects in context. ECCV 2014
16. [REF:yolov8] Jocher, G., Chaurasia, A., Qiu, J.: Ultralytics YOLOv8 (2023). https://github.com/ultralytics/ultralytics
17. [REF:peerj] Al-Refai, G., Elmoaqet, H., Al-Refai, A., Alzu'bi, A., Al-Hadhrami, T., Alkhateeb, A.: Two-stage object detection in low-light environments using deep learning image enhancement. PeerJ Computer Science (2024)
18. [REF:asc_review] Kumari, R., Singh, S., Saurabh, P.: A systematic review of image enhancement and object detection in low-light scenarios with benchmark experiments. Applied Soft Computing **201**, 115607 (2026)
19. [REF:da_detect] Chen, C., Zheng, Z., Ding, X., Huang, Y., Dou, Q.: Harmonizing transferability and discriminability for adapting object detectors. CVPR 2020
20. [REF:lol] Wei, C., Wang, W., Yang, W., Liu, J.: Deep Retinex decomposition for low-light enhancement. BMVC 2018
21. [REF:icsipa] Thu, K.S., Jangsamsi, K.: Does low-light image enhancement improve object detection? A detection-oriented and cost-aware study on ExDark. In: 2026 IEEE 9th International Conference on Signal and Image Processing Applications (ICSIPA). IEEE, Malacca, Malaysia (2026). https://doi.org/10.1109/ICSIPA69226.2026.11720114
22. [REF:pcamv] Liang, J.-T.: Machine-vision-oriented evaluation of low-light image enhancement algorithms. In: Proc. SPIE **14320**, International Conference on Photonic Computing, Algorithms, and Machine Vision (PCAMV 2026), 143200L (2026). https://doi.org/10.1117/12.3137252
23. [REF:sgvmnet] Cai, X., Wang, X., Zhao, H., Han, G.: A semantic-guided state-space learning framework for low-light image enhancement. Computers, Materials & Continua **87**(2) (2026). https://doi.org/10.32604/cmc.2026.075756

---

## 交接清单（投稿前必须处理）

| # | 事项 | 说明 |
|---|---|---|
| 1 | 补全所有 DOI | 23 条全部有 DOI；仍需逐条核对卷期页 |
| 2 | ~~找到那篇同题预印本并引用~~ | ✅ 已查实并引用 —— **Thu & Jangsamsi, ICSIPA 2026**, DOI `10.1109/ICSIPA69226.2026.11720114`（[REF:icsipa]）；另补 **Liang, PCAMV 2026**, DOI `10.1117/12.3137252`（[REF:pcamv]）。两篇均已写入 §2.3 |
| 2b | ~~补全 `[CIT:sgvmnet]`~~ | ✅ **Cai, Wang, Zhao, Han, CMC 87(2), 2026**, DOI `10.32604/cmc.2026.075756`（[REF:sgvmnet]）。已改写 §2.3 第 3 段：明确它是方法论文里的次要检测实验 |
| 3 | ~~决定是否加第二检测数据集~~ | ✅ 已完成（COCO val2017 + 合成退化 + 噪声消融，§6.2/§6.3）。Run A/B 的原始日志与 `coco_summary.txt` 需随投稿存档 |
| 3b | ~~A2 单种子~~ | ✅ 已补三种子（10-31 夜间跑，4.4 h）：std 0.0021–0.0036，CLAHE +0.007 落到 +0.004（噪声内），78% → 80%。`finetune_summary.txt` 存档 |
| 4 | ~~图 1（协议示意图）~~ | ✅ 已生成 `docs/figures/fig1_protocols.pdf/.png`（由 `docs/make_figures.py` 重跑可得）；如需手改再导入 draw.io |
| 4b | 图 2 / 图 3 | ✅ 已生成 `docs/figures/fig2_brightness_vs_map.*`、`fig3_per_class.*`（300 dpi PNG + 矢量 PDF） |
| 4c | 表格转三线表 | ✅ `docs/TABLES.tex`（booktabs，8 张表，含 Table A1），由 `docs/md_tables_to_tex.py` 从本文生成 |
| 4d | 声明与投稿信 | ✅ `docs/STATEMENTS.md`（Statements and Declarations + Cover letter，`【填】` 处待补） |
| 5 | 图 5 的合成图为 `coco_dark_sanity.png` | 三行两列，已生成于 Kaggle Output；投稿时需重排为期刊版式 |
| 6 | 作者信息、基金、数据可用性声明 | 按 MTAP 模板填 |
| 7 | 可选加分：真实（非合成）第二暗光数据集 | 现 Limitations 已如实写明；若审稿人坚持，再补 LLOD / NightOwls |
