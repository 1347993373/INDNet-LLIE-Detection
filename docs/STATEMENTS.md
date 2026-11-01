# Statements, Declarations & Cover Letter

> 用途：投稿 MTAP 时，**Statements and Declarations** 整段贴到正文末尾（参考文献之后、
> 图表之前，按 Springer 模板位置）；**Cover letter** 单独作为投稿信上传。
> `【填】` 标记处需要峰哥补真实信息。

---

## 1. Statements and Declarations（正文末尾，英文）

### Funding

The authors received no specific funding for this work.

### Competing Interests

The authors have no relevant financial or non-financial interests to disclose.

### Ethics Approval

Not applicable. This study analyses publicly available image datasets and does not
involve human participants or animals.

### Consent to Participate / Consent for Publication

Not applicable.

### Data Availability

All datasets used in this study are publicly available. ExDark is available from the
benchmark's official distribution (Loh and Chan, 2019) and was accessed through its
Kaggle mirror `hongngnguyn/exdarkdataset`; COCO val2017 (Lin et al., 2014) was accessed
through the Kaggle mirror `bardiaardakanian/mmsample`. No new images were collected.
The synthetic low-light degradation is fully specified in §6.2 of the manuscript
(illumination field, γ = 1.6, signal-dependent noise, luminance calibration) and is
implemented in the released code, so every degraded image can be regenerated
bit-exactly from the seed recorded in the script.

### Code Availability

The evaluation pipeline (Protocol I, Protocol II-A/II-B, the COCO degradation and the
noise ablation) and the trained weights of the in-house restoration network are released at
【填：GitHub 地址】 and archived at 【填：Zenodo DOI】.

### Author Contribution Statement

【填：按 CRediT 写，例如】
Conceptualization: A.B.; Methodology: A.B., C.D.; Software: A.B.; Validation: C.D.;
Formal analysis: A.B.; Investigation: A.B., C.D.; Data curation: C.D.;
Writing – original draft: A.B.; Writing – review & editing: A.B., C.D.;
Visualization: A.B.; Supervision: 【通讯作者】; Project administration: 【通讯作者】;
Funding acquisition: 【通讯作者】.

### Acknowledgements

The authors thank 【填：如有】 for helpful discussions, and the Kaggle platform for
providing the GPU resources used for all experiments.

---

## 2. Cover Letter（单独文件上传）

Dear Editor-in-Chief,

We submit for consideration in *Multimedia Tools and Applications* the manuscript
**"Beyond PSNR: A Two-Dataset Analysis of Low-Light Image Enhancement for Object
Detection"**.

Low-light image enhancement (LLIE) is routinely inserted as a pre-processing step before
object detectors, on the assumption that a clearer image must be a better detection input.
Our paper tests that assumption under a controlled protocol in which only the input
changes, and asks not merely *whether* enhancement helps but *why* it usually does not.

The manuscript reports four results that we believe are of interest to the journal's
readership:

1. **The negative result replicates across datasets and detectors.** Twelve operators on
   ExDark and six conditions on a controlled COCO degradation (twenty-three
   condition–recipe rows in total) are scored with a frozen detector; no enhancement
   reliably improves mAP, and the ordering reproduces with a second detector.
2. **Pixel-fidelity metrics do not predict detection.** Because the COCO benchmark
   retains the clean image, restoration fidelity can be measured directly rather than
   inferred. One operator gains 2.82 dB PSNR and 0.166 SSIM while *losing* 0.020 mAP;
   two learned methods 0.4 dB apart in PSNR differ by 0.024 mAP; within the degraded
   branch the PSNR–mAP correlation is r = 0.007.
3. **A mechanism-level decomposition.** Ablating sensor noise halves the damage and flips
   CLAHE from harmful to neutral; local over-exposure is the only statistic that behaves
   consistently in the noiseless setting; the remainder is domain shift.
4. **A two-protocol decomposition with error bars.** Re-training the detector is worth
   +0.148 mAP, and 80% of the residual loss of learned enhancement is domain shift rather
   than restoration quality — measured over three training seeds (std 0.0021–0.0036), so
   no conclusion rests on a single run.

We are aware of, and cite, two recent 2026 studies that reach a similar conclusion on
ExDark through cost-oriented evaluation [Thu & Jangsamsi, ICSIPA 2026; Liang, PCAMV 2026].
Our contribution is complementary rather than duplicative: those works establish *that*
the assumption fails and at what computational price, whereas we provide the quantitative
account of *why* — a ground-truth-based fidelity analysis, a noise/exposure ablation, and
the domain-shift decomposition — plus a stress test showing that the brightness relation
fitted on ExDark does not transfer to a second dataset. We also discuss a report of small
positive gains under a much more heavily adapted detector and reconcile it with our
measurements.

The manuscript is original, has not been published elsewhere, and is not under
consideration by another journal. All authors have approved the submission and declare no
competing interests. All datasets are public and the code is released.

We hope the work is suitable for *Multimedia Tools and Applications*, whose readership
combines multimedia signal processing with applied computer vision — the two halves this
paper tries to connect.

Yours sincerely,

【填：通讯作者姓名, 职称, 单位, 邮箱】
on behalf of all authors

---

## 3. 投稿前自查（对照 Springer 要求）

| 项 | 状态 |
|---|---|
| 正文含 Statements and Declarations 段 | 贴第 1 节 |
| 图件单独文件、≥300 dpi、编号与正文一致 | `docs/figures/` 里的 PDF/PNG；Fig. 4/5 来自 Kaggle Output |
| 表格可编辑（非图片） | 用 `docs/TABLES.tex`（LaTeX）或在 Word 里重排 |
| 参考文献体例统一（期刊名斜体） | 23 条已带 DOI，模板里统一斜体化 |
| 作者 ORCID / 通讯邮箱 | 【填】 |
| 推荐审稿人（可选） | 建议避开 ICSIPA/PCAMV 作者，避免利益冲突 |

---

## 4. 代码开源与 DOI（GitHub + Zenodo）

**在哪建：github.com 官网，不是 Kaggle。**
Kaggle 只能挂数据集和 notebook，没有仓库概念、不能 clone、也不会发 DOI；它是我们跑实验的
运行时环境，和"对外发布代码"是两件事。`indnet-code` 数据集继续留着跑实验，不用动。

### 4.1 仓库里放什么

| 放 | 不放 |
|---|---|
| `models/ losses/ datasets/ utils/ train.py test.py` | `data/`（ExDark/LOL 原始图，体积大且有版权） |
| `configs/`、`scripts/` | `*.tar.gz`（打包产物） |
| `notebooks/`（五个 Kaggle 单元 + 复现入口） | `runs/`、`__pycache__/`、`*.pyc` |
| `docs/`（本目录：论文稿、图表、TABLES.tex） | 权重 `*.pth`（走 Zenodo，见 4.3） |
| `README.md` + `requirements.txt` + `LICENSE`（建议 MIT） | 任何带路径的本机配置文件 |

README 最少写四段：环境依赖、如何复现 Table 1（跑哪个 notebook 单元）、如何复现 Protocol II、
权重下载地址（Zenodo DOI）。

### 4.2 三步发上去

**方式 A：GitHub Desktop（推荐，图形界面）**

1. github.com 注册/登录 → 右上 `+` → New repository → 名字 `INDNet-LLIE-Detection` → **Public** →
   勾 `Add a README file` → Create
2. 装 GitHub Desktop → `File → Clone repository` → 选刚建的仓库 → 选本地目录
3. 把 `INDNet/` 里该放的文件夹拖进克隆目录 → 填一句 Summary → `Commit to main` → `Push origin`

**方式 B：命令行**

```bash
cd D:/WorkBuddy/2026-10-13-15-34-22/INDNet
git init
git add models losses datasets utils configs scripts notebooks docs train.py test.py README.md
git commit -m "Release code and evaluation pipeline for the LLIE-detection study"
git branch -M main
git remote add origin https://github.com/<你的用户名>/INDNet-LLIE-Detection.git
git push -u origin main
```

`.gitignore` 至少这几行：

```
data/
runs/
__pycache__/
*.pyc
*.pth
*.tar.gz
```

（如果用命令行的 `git add` 逐项添加，其实不写 .gitignore 也不会误传；但以后更新容易漏，建议写。）

### 4.3 拿 DOI（Zenodo）

1. 用 **GitHub 账号**登录 zenodo.org → 右上头像 → `Settings` → `GitHub`
2. 找到 `INDNet-LLIE-Detection` → 把开关拨到 **ON**（授权一次即可）
3. 回 GitHub → `Releases` → `Create a new release` → Tag 填 **v1.0.0** → 标题填
   *Code for "Beyond PSNR"* → Publish release
4. 等 1–2 分钟，Zenodo 的 GitHub 页面会出现一条记录，带 **DOI**（形如 `10.5281/zenodo.xxxxxxx`）
5. 把 DOI 填回 `STATEMENTS.md` 的 Code Availability，以及正文

权重（三个 `best.pth`，共约 45 MB）建议**不放进 git**，而是随 Zenodo 的 release 一起上传：
在 Zenodo 那条记录的 `Edit` → `Files` → `Upload` 里加一个 `weights.zip`，然后 `Publish` 新版本。

### 4.4 国内网络的两个备选

| 问题 | 备选 |
|---|---|
| github.com 打不开/很慢 | 代码镜像到 **Gitee（码云）**，但**投稿链接仍写 GitHub + DOI** —— DOI 在任何网络下都能打开，是最稳的引用方式 |
| Zenodo 打不开 | 改用 **Figshare**（figshare.com）：直接上传 `INDNet-code-v1.0.0.zip`，它同样发 DOI，不依赖 GitHub |

**关键点：审稿人真正需要的是一个永久 DOI，GitHub 只是中间环节。** 如果 Zenodo/Figshare 都不顺，
最低成本方案是 Figshare 传 zip → 拿 DOI → 正文写 DOI，GitHub 作为附加链接。
