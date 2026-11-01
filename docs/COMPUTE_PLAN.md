# 算力方案：本地 vs 云（实测定论）

> 本文结论基于 2026-10-13 在本机的**实际基准测试**，不是估算。

---

## 一、本机硬件实况

| 项目 | 实测值 | 评价 |
|---|---|---|
| CPU | AMD64 Family 23 Model 104（Zen2，8 核 16 线程） | 够用，多线程利用充分 |
| 内存 | 总 15.3 GB，**可用仅 4.2 GB** | ⚠️ 偏紧，是主要瓶颈之一 |
| GPU | **无 NVIDIA 独显**（集显），`torch.cuda.is_available()` = False | ❌ 无法 GPU 加速 |
| 磁盘 | C 盘可用 14.7 GB / D 盘可用 51.0 GB | 数据够放，注意别写 C 盘 |

---

## 二、CPU 训练实测（200 轮，LOL-v1 485 对）

在 `torch.set_num_threads(16)` 下实测「前向 + 反向 + 优化器更新」单次迭代耗时：

| 配置 | s/iter | min/epoch | **200 轮总计** |
|---|---|---|---|
| 当前配置 bc32 / patch256 / batch8 | 6.68 | 6.68 | **22.3 小时** |
| 降 batch 到 4 | 3.21 | 6.48 | 21.6 小时 |
| 降 patch 到 192 | 3.62 | 3.62 | 12.1 小时 |
| 降通道数到 16 | 2.79 | 2.79 | 9.3 小时 |
| 轻量配置 bc16 / patch128 | 0.74 | 0.74 | **2.5 小时** |

**参考**：Kaggle 免费 T4 按 25× 加速估算，当前配置跑完 200 轮约 **0.9 小时**。

---

## 三、结论：分三种情况

### ✅ 本地完全没问题

- 数据处理、代码调试、跑冒烟测试（`--smoke`）
- **P1 基线复现**——注意这里只需要**推理**，不需要训练。用 Retinexformer 的预训练权重在 15 张测试图上跑一遍，CPU 上几分钟就完事
- 论文图表制作、可视化
- 小规模调参探索（用轻量配置，2.5 小时/次）

### ⚠️ 本地可行但慢

- 正式训练 INDNet 主模型：**22 小时/次**
- 而实验计划需要大约 **9 次训练**（主实验 ×2 数据集 + 消融 4 组 + 损失权重敏感性 3 组）
- 即 **连续不关机跑 8 天左右**。可行，但要考虑：笔记本散热降频、4.2 GB 可用内存偏紧、跑满一周时间

### ❌ 不推荐的本地做法

为了在本地跑得动而把配置砍到 `bc16 / patch128`（2.5 小时/次）——
**这会让你的 PSNR 明显低于 SOTA**，而 PAA 明确要求"与 SOTA 在同一条件下对比"。
为了省算力牺牲指标，得不偿失。

---

## 四、推荐方案：本地调试 + Kaggle 免费 GPU 训练

**这是本项目的最优解，零成本。**

### 为什么是 Kaggle 而不是租 GPU

1. **免费额度够用**：每周 **30 GPU 小时**（T4×2 或 P100 16GB），本机跑一次 22 小时的训练，在 T4 上约 1 小时——**一周能跑 30 次**，整个项目 10–15 次训练绰绰有余
2. **数据集本来就在 Kaggle 上**：LOL 和 ExDark 都是 Kaggle 公开数据集，可以**直接挂载到 `/kaggle/input`，零下载成本**。这比 AutoDL 还要方便
3. **不用花钱**：省掉之前估算的 ¥160–240

### 具体操作

1. 注册 kaggle.com，**完成手机验证**（不验证拿不到 GPU 加速器）
2. 新建 Notebook → 右侧 `Settings` → `Accelerator` 选 **GPU T4 ×2**
3. `Add Input` → 搜索并挂载：
   - `soumikrakshit/lol-dataset`（LOL-v1）
   - ExDark 对应数据集
   挂载后位于 `/kaggle/input/<数据集名>/`（只读）
4. 上传代码：把 INDNet 项目打包上传，或用 Kaggle API 从命令行 push
5. 训练时用 **`Save & Run All`（Commit）** 后台执行，关掉浏览器也不会中断
6. **checkpoint 存到 `/kaggle/working/`**（20 GB 自动保存）。
   `train.py` 已内置 `save_every` 与 `best.pth` 机制，正好适配 Kaggle 的 9–12 小时会话上限

### 关键适配

Kaggle 上数据集是**只读**的，所以配置里 `data.root` 指向 `/kaggle/input/...`，
而 `output_dir` 要指到 `/kaggle/working/`：

```yaml
data:
  root: /kaggle/input/lol-dataset/lol_dataset
model:
  base_channels: 32          # T4 16GB，用完整配置没问题
  illumination_branch: true
  noise_branch: true
train:
  batch_size: 16             # T4 显存充足，可以开大
  num_workers: 2             # Kaggle 只有 4 核，别设太大
  epochs: 200
  save_every: 25             # 会话可能被打断，保存密一点
experiment:
  output_dir: /kaggle/working/INDNet_LOLv1
eval:
  save_dir: /kaggle/working/results/LOLv1
```

---

## 五、备选方案对比

| 方案 | 成本 | 单次训练耗时 | 评价 |
|---|---|---|---|
| **Kaggle 免费 GPU** | ¥0 | ~1 小时 | ✅ **首选**，额度充足且数据集现成 |
| 本地 CPU（当前配置） | ¥0 | ~22 小时 | ⚠️ 可行，但一周不能关机 |
| Colab 免费 | ¥0 | ~1 小时 | 会话易断、GPU 不保证，作为 Kaggle 的备用 |
| AutoDL 租 4090 | 约 ¥160–240 | ~0.6 小时 | 最省时间，想快速推进再考虑 |

**建议节奏**：先在本地把代码和数据流程全部调通（零成本），
再上 Kaggle 跑正式训练。两边用同一个代码库，只改配置文件。
