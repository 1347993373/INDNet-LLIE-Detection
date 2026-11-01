# 云 GPU 部署指南

本机无 NVIDIA 独显，训练全部放在云上进行。本文给出可直接照做的方案。

---

## 一、平台选择

| 平台 | 特点 | 适合场景 |
|---|---|---|
| **AutoDL**（推荐） | 国内，按小时计费，4090 约 ¥2.0/h，3090 约 ¥1.5/h，支持"无卡模式"开机配置环境 | 本项目首选，性价比最高 |
| 恒源云 / 矩池云 | 与 AutoDL 类似，价格接近 | 备选 |
| 腾讯云 / 阿里云 | 稳定但贵，按量计费常为 ¥5–15/h | 学校有经费时用 |
| Colab Pro | 便宜，但会话会断、数据不持久 | 只适合临时试验 |

**算力需求评估**：LOL 只有 485 对 400×600 的图，INDNet 参数量约 2–5 M。
单张 **RTX 3090 / 4090（24GB）绰绰有余**，甚至 12GB 的 3060 也够。
不要租多卡，本项目用不上。

---

## 二、实例配置

- **GPU**：RTX 4090（24GB）× 1 —— 训练快、迭代调试省时间，贵一点但总成本更低
- **镜像**：`PyTorch 2.1.0 / Python 3.10 / CUDA 12.1`（平台上直接选，不要自己从零装）
- **系统盘/数据盘**：数据盘至少 60 GB（LOL 350MB + ExDark 几 GB + 输出图，余量充足）

> **关键提醒**：AutoDL 的 `/root/autodl-tmp` 是数据盘，**关机不丢失**；
> 系统盘 `/root` 之外的改动可能被重置。数据集和实验输出一律放 `autodl-tmp`。

---

## 三、首次配置流程

### 1. 用「无卡模式」开机
无卡模式只收极低的费用（约 ¥0.1/h），适合装环境、传数据。
配置完再关机、切到有卡模式启动训练。

### 2. 上传代码

```bash
# 本地打包（在 INDNet 目录下执行）
tar --exclude='.venv' --exclude='data' --exclude='experiments' -czf indnet_code.tar.gz .

# 上传（用平台的 JupyterLab 拖拽，或 scp / 阿里云盘中转）
# 服务器上解压到数据盘
mkdir -p /root/autodl-tmp/INDNet
tar -xzf indnet_code.tar.gz -C /root/autodl-tmp/INDNet
cd /root/autodl-tmp/INDNet
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
# 云镜像通常已自带匹配 CUDA 的 torch，不要重复安装覆盖它
# 若确实缺 torch：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. 下载数据

```bash
mkdir -p data && cd data

# LOL-v1（约 350 MB）—— 优先用 Kaggle 或 OpenXLab，国内直连较快
# 方式 A：Kaggle CLI（需先配置 ~/.kaggle/kaggle.json）
kaggle datasets download -d soumikrakshit/lol-dataset && unzip -q lol-dataset.zip

# 方式 B：OpenXLab
pip install openxlab
openxlab dataset get --dataset-repo OpenDataLab/LOL

# ExDark（下游检测用）
kaggle datasets download -d kushagratandon12/dataset-for-object-detection-in-low-light
unzip -q dataset-for-object-detection-in-low-light.zip -d ExDark

cd ..
python scripts/check_data.py --root data    # 校验
```

### 5. 冒烟测试（必做）

**在有卡模式下先跑这一步**，确认数据加载、前向、反向、优化器全部正常：

```bash
python train.py --config configs/lol_v1.yaml --smoke
```

看到 `冒烟测试完成` 字样即可开始正式训练。

---

## 四、正式训练

```bash
# 后台运行，日志落盘，断开 SSH 也不会中断
nohup python train.py --config configs/lol_v1.yaml > train_stdout.log 2>&1 &

# 实时查看
tail -f train_stdout.log

# 查看 GPU 占用
nvidia-smi
```

**显存不够时的调整**（改 `configs/lol_v1.yaml`）：

| 显存 | batch_size | patch_size |
|---|---|---|
| 6 GB | 4 | 192 |
| 8 GB | 8 | 256 |
| 12 GB | 16 | 256 |
| 24 GB | 32 | 320 |

**注意**：改 batch_size 后学习率要同步调整（大致线性缩放规则），
且**论文里报告的配置必须与实际训练一致**。

---

## 五、P1 阶段要复现的基线

P1 是本项目最关键的关卡，需要在云上跑通 Retinexformer 拿到基准数字。

| 基线 | 仓库 / 权重 |
|---|---|
| **Retinexformer**（必做，主基线） | `github.com/caiyuanhao1998/Retinexformer`，README 提供 LOL-v1 预训练权重 |
| RetinexMamba | `github.com/...` 搜索 "RetinexMamba"，2024 年工作 |
| HVI（CVPR'24） | `github.com/Fediory/HVI-CIDNet` |
| SCI | `github.com/vis-opt-group/SCI`，极轻量，适合验证评测管道 |
| Zero-DCE++ | `github.com/Li-Chongyi/Zero-DCE_extension` |

**通过标准**：Retinexformer 在 LOL-v1 eval15 上 PSNR ≥ 25.0、SSIM ≥ 0.84（±0.2）。
达不到就停下排查，常见原因见 `docs/ROADMAP.md` 的 P1 章节。

---

## 六、成本估算

| 阶段 | 卡时 | 费用（4090 @ ¥2/h） |
|---|---|---|
| P0 环境与数据 | 2–3 h | ¥5 |
| P1 基线复现 | 5–8 h | ¥15 |
| P2 主方法调参 | 30–50 h | ¥60–100 |
| P3 完整实验（主表 + 消融 + 应用） | 40–60 h | ¥80–120 |
| **合计** | **约 80–120 卡时** | **约 ¥160–240** |

**省钱要点**：

1. 不需要跑的时候**立刻关机**，不要只断开 SSH
2. 配置环境和传数据用**无卡模式**
3. 调参阶段用`--smoke`或小 epoch 数验证想法，别一上来就 200 epoch
4. 每次训练完把 `experiments/` 和 `results/` 同步回本地或对象存储，避免实例回收导致成果丢失
