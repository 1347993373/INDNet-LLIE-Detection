# Kaggle 免费 GPU 使用指南

**免费额度：每周 30 GPU 小时**（T4 ×2 或 P100，均为 16 GB 显存），额度每周重置。
本项目在 T4 上训练一次约 1 小时，整个论文所需实验 10–15 次，**一周额度足够跑完，成本为零**。

---

## 一、开通（一次即可）

1. 注册 [kaggle.com](https://www.kaggle.com)，用邮箱即可，**不需要信用卡**
2. **完成手机验证** —— 这是解锁 GPU 加速器的前提。
   入口：右上角头像 → `Settings` → `Phone Verification`
   > 未验证时 Accelerator 里只有 None / TPU，看不到 GPU 选项
3. 验证后在 `Settings` 页面可以看到每周 GPU 额度与剩余时间

---

## 二、把代码传上去

`/kaggle/input` 是**只读**的，代码必须作为一个 Dataset 挂载进来。

### 1. 本地打包（在 INDNet 目录下执行）

已经写好打包脚本，直接运行即可：

```bash
python scripts/pack_for_kaggle.py            # 生成 indnet-code.tar.gz
python scripts/pack_for_kaggle.py --list     # 只预览，不生成
```

实测输出：**24 个文件，压缩后 43.8 KB**。

**传什么、不传什么**（脚本用白名单机制，只装下面这些）：

| 打包 ✅ | 说明 |
|---|---|
| `train.py` / `test.py` | 训练与评测入口 |
| `configs/` | 两个配置文件 |
| `datasets/` `models/` `losses/` `utils/` | 全部核心代码 |
| `scripts/` | 数据校验与打包脚本 |
| `requirements-kaggle.txt` | Notebook 实际安装的依赖 |
| `docs/` `notebooks/` `README.md` | 体积很小，带着方便在 Kaggle 上查阅 |

| 不打包 ❌ | 原因 |
|---|---|
| `data/` | **数据集在 Kaggle 上挂载即可**，传上去纯属浪费 |
| `experiments/` | 训练权重和日志，体积大且每次重跑都会变 |
| `results/` | 评测输出图像，同样不必上传 |
| `__pycache__/` | 字节码缓存，无用 |

> 为什么用白名单而不是 `tar --exclude`？
> 黑名单容易漏——比如哪天在项目里放了个临时目录或压缩包，`tar .` 会一并打进去。
> 白名单则保证**只有明确列出的代码会出去**，不可能误传数据集或个人文件。

### 2. 上传为 Dataset

1. Kaggle 首页 → `Datasets` → `New Dataset`
2. 把 `indnet-code.tar.gz` 拖进去
3. 标题填 **`indnet-code`**（Notebook 里就按这个名字找）
4. 可见性选 **Private**（私有，别人看不到）
5. 点 `Create`

> Kaggle 会自动解压，挂载后即为扁平目录（解压到 `/kaggle/input/indnet-code/`）。

**以后改了代码怎么办**：回到这个 Dataset 页面点 `New Version` 重新上传即可，
Notebook 里重新挂载就会拿到新版本 —— 不需要每次重传整个项目。

---

## 二·五、要挂载的三个输入

在 Notebook 的 `Input` → `Add Input` 里依次添加（**搜索框支持直接粘贴 URL，这是最稳的方式**）：

| 用途 | 数据集 | URL / 搜索词 |
|---|---|---|
| 你的代码 | `indnet-code` | 自己上传的那个私有数据集 |
| 增强实验 | LOL-v1 | `https://www.kaggle.com/datasets/soumikrakshit/lol-dataset` |
| 检测实验 | ExDark | `https://www.kaggle.com/datasets/hongngnguyn/exdarkdataset` |

> **ExDark 的坑**：直接搜 "ExDark" 会返回几十个**别人训练好的 YOLO 权重**，
> 那些不是原始数据集，挂上去也找不到标注。要用上面那个 URL，
> 认准描述里的 **7,363 low-light images** 与 **12 object classes**。

---

## 三、创建 Notebook 并做三项设置

`Create` → `New Notebook`，然后**必须改三个地方**，否则跑不起来：

| 位置 | 设置项 | 值 | 不改的后果 |
|---|---|---|---|
| 右侧 `Settings` | `Accelerator` | **GPU T4 x2** | 纯 CPU 跑，白等几小时 |
| 右侧 `Settings` | `Internet` | **On** | pip 装不了依赖 |
| 右侧 `Input` | `Add Input` | 挂载 `indnet-code` 与 LOL 数据集 | 找不到代码或数据 |

LOL 数据集搜索 **`soumikrakshit/lol-dataset`** 添加。

---

## 四、导入我准备好的 Notebook

我已经写好 `notebooks/kaggle_train.ipynb`，包含了完整流程，直接导进去用：

Notebook 编辑器左上角 `File` → `Import Notebook` → 上传该文件。
（或者把里面的单元格内容手动复制过去也行。）

它包含 10 个步骤：

| 步骤 | 作用 |
|---|---|
| 1 | 环境自检——确认 GPU 真的挂上了 |
| 2 | 查看挂载了哪些数据集 |
| 3 | 把代码从只读区复制到可写工作区 |
| 4 | 安装依赖（**只装缺失的包，绝不动 torch**） |
| 5 | 生成运行时配置，自动识别数据路径 |
| 6 | 数据校验——必须看到 485 / 15 配对 |
| 7 | 冒烟测试——验证管道 |
| 8 | 正式训练 |
| 9 | 评测并导出分解图 |
| 10 | 打包结果供下载 |

**关键设计**：第 4 步用的是 `requirements-kaggle.txt` 而不是 `requirements.txt`。
因为 Kaggle 镜像预装了**与 CUDA 匹配的 torch**，一旦执行 `pip install -r requirements.txt`
就会把 CUDA 版覆盖成 CPU 版，GPU 白开。

---

## 五、运行

### 调试阶段
逐个单元格执行（`Shift + Enter`），看清每一步输出。

### 正式训练
全部验证通过后，点 **`Save Version`** → 选 **`Save & Run All`**。

这个模式会在一个**全新的干净会话里从头到尾完整执行**，**关掉浏览器也不会中断**，
最多可运行 12 小时。训练完成后结果自动保存。

> `Save & Run All` 就是旧版的 `Commit` 行为。

### 注意
交互式调试时，**闲置 20 分钟会话会被回收**。所以正式训练一定要用 `Save & Run All`，
不要挂着一个交互会话干等。

---

## 六、结果保存与下载

- 所有产物写到 **`/kaggle/working`**（上限 20 GB，随 Notebook 版本自动保存）
- Notebook 第 10 步会把训练产物打包成 `indnet_outputs.zip`
- 在右侧 `Output` 面板可直接下载

**跨 Notebook 复用**：在另一个 Notebook 里 `Input → Add Input`，
搜索**你自己这个 Notebook 的名字**，就能把它的输出当作输入挂进来 —— 比如训练 Notebook 的输出
（`best.pth` + 增强结果图）直接给评测 Notebook 用，不用下载再上传。

---

## 七、中断了怎么续训

`train.py` 每 `save_every` 轮会存一次 checkpoint（Kaggle 配置里设为 25 轮）。
会话被回收后，新建 Notebook 挂上之前的输出，然后：

```python
!{sys.executable} train.py --config {RUN_CFG} \
    --resume /kaggle/working/INDNet_LOLv1/last.pth
```

---

## 八、额度管理要点

| 事项 | 说明 |
|---|---|
| 额度 | 每周 30 GPU 小时，**周一重置** |
| 查看 | 右上角头像 → Settings，或 Notebook 内 Accelerator 旁的剩余提示 |
| 省额度 | **不训练时把 Accelerator 切回 `None`**，就不会消耗 GPU 时长 |
| 排队 | 高峰期 GPU 实例可能排队，稍等或换时段 |
| 单次上限 | 12 小时（TPU 为 9 小时） |

---

## 九、常见问题

**Q：Accelerator 里看不到 GPU 选项？**
手机没验证。去 `Settings → Phone Verification` 完成验证。

**Q：`pip install` 报网络错误？**
右侧 `Settings → Internet` 没打开。

**Q：报 `CUDA out of memory`？**
把配置里 `data.batch_size` 从 16 降到 8（T4 是 16 GB，通常够用，OOM 多半是别的原因）。

**Q：Add Input 面板显示「No Data Sources Found」？**
这是**正常的**——搜索框是空的，面板不会自动列出所有数据集。三种解决办法：

1. **最快**：在搜索框里贴**数据集链接**（占位文字写明支持 URL），
   例如 `https://www.kaggle.com/datasets/soumikrakshit/lol-dataset`
2. 点 **`Your Work`** 筛选标签，只看你自己的数据集和 Notebook
3. 直接输关键词，如 `lol dataset`、`indnet`

> 找不到自己刚创建的私有数据集时，多半是索引还没刷新，等 1 分钟刷新页面即可。
> 链接获取方式：打开数据集页面，从浏览器地址栏复制。

**Q：找不到 `/kaggle/input/indnet-code`？**
先确认已挂载；若挂载了但名字不同也没关系——Notebook 第 3 格会按特征自动定位代码包，
不依赖数据集名称。

**Q：训练太慢？**
检查第 1 步输出的 `CUDA : True`。如果是 `False`，说明跑在 CPU 上，检查 Accelerator 设置。

**Q：额度用完了怎么办？**
每 30 小时约 30 次训练，正常项目够用。真不够可以：
- 等下周重置
- 用 Google Colab 免费 T4 作为备用（会话易断，需频繁 checkpoint）
- 临时租 AutoDL（约 ¥1.5–2/小时）

---

## 七、推荐做法：一条链路跑完（含消融 + 检测）

### 为什么要有这个入口

Kaggle 会话**随时会被回收**，并且会**清空 `/kaggle/working`**。
我们已经在真实环境中踩过两次：

| 现象 | 后果 |
|---|---|
| 训练好的 `best.pth` 随会话消失 | 检测评测时"缺少权重"，只能评测原图，白跑半小时 |
| 逐个格子手动跑 | 顺序错、中间断了不知道从哪接着跑 |

所以现在提供一个**单一入口**：`notebooks/kaggle_pipeline.py`。
它做三件事，且每一步都幂等：

1. **算力自检** —— 没有 GPU 就直接停下并给出开启指引（避免 CPU 上跑 3 天）
2. **补齐消融权重** —— 缺哪个变体就训哪个，已存在则跳过
3. **ExDark 检测评测** —— 输出 4 行 mAP 对比表

### 用法

在 Kaggle Notebook 里新建一格，粘贴 `notebooks/kaggle_pipeline.py` 的全部内容即可。

想改行为，在文件里 `main()` 之前加环境变量：

```python
os.environ["EXDARK_LIMIT"]    = "1500"   # 检测只跑前 1500 张，快速验证
os.environ["RETRAIN"]         = "1"      # 强制重训消融变体
os.environ["ALLOW_CPU_TRAIN"] = "1"      # 允许 CPU 训练（极慢，默认禁止）
```

### 必须用 `Save Version -> Save & Run All`

这是解决"权重又丢了"的根本办法：

- **交互式运行**：产物只存在于当前会话，会话一回收就没了
- **`Save & Run All`**：在干净会话里从头跑一遍，产物**随版本持久保存**

所以正式跑的时候：

1. 先设成 `EXDARK_LIMIT=1500`，`Save & Run All` 跑一遍确认全通（约 10 分钟）
2. 再把 `EXDARK_LIMIT` 去掉，重新 `Save & Run All` 跑全量（约 3.5 小时）

跑完在 Notebook 的 **`Output`** 标签里能下载 `ablations/` 和 `detection/`。
下次会话重启时，把这个版本的 Output 作为 Input 挂回来，就不必重训。

### 三个环境变量速查

| 变量 | 作用 |
|---|---|
| `EXDARK_LIMIT` | 检测评测的图像数上限，0 = 全部 |
| `EXDARK_CLASS_MODE` | 手动指定 class 列编码：`id0` / `id1` / `name`（平时自动判定） |
| `EXDARK_ALLOW_NO_ENHANCER` | 允许在没有增强权重时只评测原图（默认禁止） |
| `RETRAIN` | 强制重训消融变体 |
| `ALLOW_CPU_TRAIN` | 允许 CPU 训练（默认禁止） |

---

## 八、`Save & Run All` 之前必须做的清理（血泪教训）

### `Save & Run All` 会执行**整个 Notebook 的所有单元格**

它不是在当前会话接着跑，而是**新开一个干净会话，从第 1 格跑到最后一格**。
所以如果这个 Notebook 里还留着之前调试时粘的旧格子，它们会被**一起重跑**，后果很实际：

| 残留的旧格 | 重跑后会发生什么 |
|---|---|
| 旧的训练格 | 又训 200 轮，白烧约 1 GPU 小时 |
| 旧的消融格（非幂等版本） | **删掉已有权重**再训三个变体，白烧约 3 GPU 小时 |
| 旧的损失补丁格 | 重新往 `losses.py` 写补丁文本（新版代码已内置，属重复操作） |
| 旧的检测格（未修正版本） | 用**旧的类别映射**重算一遍，类别错位，结果不可用 |
| 任意旧格里的 `FORCE_REFRESH` | 反复删除并重建 `/kaggle/working/INDNet`，浪费时间 |

最坏情况：额度被烧掉大半，产物还是**新旧混杂**的，无法判断哪份可信。

### 推荐做法：新建一个干净的 Notebook

1. `kaggle.com/code` → **`New Notebook`**（旧的留着当参考，别删）
2. 右侧 `Settings`：
   - `Accelerator` → **GPU T4 x2**
   - `Internet` → **On**
3. 右侧 `Input` → `Add Input`，挂三个：
   - 你自己的 **`indnet-code`**（确认是新版本）
   - **`soumikrakshit/lol-dataset`**
   - **`hongngnguyn/exdarkdataset`**
4. **只保留一个代码格**，粘贴 `notebooks/kaggle_pipeline.py` 的全部内容
5. `Save Version` → **`Save & Run All`**

> 干净 Notebook 里只有一格，`Save & Run All` 的行为就完全可预测了。

### 如果你坚持用旧 Notebook

那必须先把**除 pipeline 格以外的所有格子全部删除**（选中格子 → 右上角删除图标）。
删之前建议先 `File → Save Version` 留个快照，万一误删还能找回。

判断标准：`Save & Run All` 开始后，输出里**第一行就应该出现 pipeline 的「算力自检」**；
如果先看到别的格子的输出（比如"正在安装 ultralytics"、"已刷新代码"），说明还有残留格子。

> 开了 `RUN_FINETUNE=1` 时，横幅末尾会多出 ` -> 检测器微调评测(A2)`。
> 判据是**前缀**而不是整行 —— 只要开头是 `INDNet 一键链路：预检 -> 补齐消融权重 -> ExDark 检测评测`
> 就说明没有残留格子。

---

## 九、运行前预检（约 2 分钟，强烈建议）

正式流程一跑就是 **3–4 小时**。如果 3 小时后才发现 ExDark 没挂上、或者代码包还是旧版，代价太大。
所以先花 2 分钟跑一次预检：`notebooks/kaggle_preflight.py`。

它**不训练、不评测**，只检查五件事：

| 检查项 | 不通时说明什么 |
|---|---|
| GPU | 没选 Accelerator → 会慢几十倍 |
| LOL 数据集 | 数量必须是 **our485 485 对 / eval15 15 对** |
| ExDark 数据集 | 要有图像目录 + `annotations.csv` |
| 代码包 | 三个脚本齐全 = 数据集已更新到最新版 |
| `/kaggle/working` 可写 + Internet | 检测环节要装 ultralytics、下 yolov8n.pt |

全部 `[OK]` 时，它还会打印预计耗时。有任何 `FAIL` 会列出修复建议。

用法：在交互会话里新建一格，粘进去运行。**别放进 `Save & Run All` 的正式 Notebook 里**，
否则每次正式跑都会多花 2 分钟重复检查（无害，但没必要）。

### 预计耗时（GPU T4 单卡，实测口径）

| 阶段 | 耗时 |
|---|---|
| 消融 3 个变体 · 各 200 轮 | 约 2.5 – 3.5 小时 |
| ExDark 检测 4 种输入 × 7362 张 | 约 15 – 25 分钟 |
| 环境安装 / 数据索引 / 汇总 | 约 5 分钟 |
| **合计** | **约 3 – 4 小时** |

> ⚠️ `EXDARK_LIMIT` 只能缩短**检测**那一段，**不能缩短训练**。
> 所以在权重还没训出来时，"快速验证"照样要花 3 小时——这正是要先跑预检的原因。

---

## 十、常见界面疑问

**Q：`Save & Run All` 提交后，Logs 标签只显示「Running for 366.5s」，没有过程输出，正常吗？**

正常。Logs 标签在**运行期间**只显示一条累计时长（以及 Accelerator / Environment / Output 三项信息），
**不会流式输出 stdout**。等某一格跑完后，Logs 里才会出现该格的耗时记录，点进去才能看到它的输出。

同时 **`Output` 显示 `0 B` 也是正常的** —— 产物要等整次运行结束后才定稿。
训练过程中 `train.py` 每 25 轮往 `/kaggle/working/ablations/.../` 写一次 checkpoint，
但这些文件在运行结束前不会反映到 Output 的大小上。

所以判断"跑得好不好"只有一个办法：**等它跑完，再看每一格的输出**。
想提前确认环境没问题，就在交互会话里先跑 `notebooks/kaggle_preflight.py`（约 2 分钟）。

**Q：运行期间能关掉浏览器吗？**

能。`Save & Run All` 是在后台的独立会话里执行的，关掉浏览器、关掉电脑都不影响。
但**不要点 Edit 回到编辑器去手动操作这个版本**，也不要重启会话 —— 会让这次运行中断。

**Q：Accelerator 显示 `GPU T4 ×2` 就说明 GPU 用上了吗？**

基本是。但代码只使用第 0 块 GPU。真正的判据是运行输出里的 `CUDA` 与显卡名
（pipeline 第 1 步会打印 `GPU: Tesla T4 | 15.0 GB`）。

---

## 十一、数据集挂载失败（ERRORED_MOUNTING_DATASET）

**现象**：运行一开始就报

```
mount data: ERRORED_MOUNTING_DATASET: DataSourceUrl.addDataset(...):
retry budget exhausted (30 attempts): rpc error: code = Internal desc = dataset loading failed
```

**这是 Kaggle 平台侧的挂载失败，不是你的操作问题，也不是代码问题**（平台重试 30 次都没成功）。
处理顺序（按有效性排序）：

1. **刷新页面 → 右侧 `Input` 里删掉该数据集 → 重新 `Add Input` 加回来**（最常见有效）
2. 仍失败 → 右上角 **`Restart Session`**，然后重新运行
3. 还失败 → 到该数据集页面点 **`New Notebook`**，让 Kaggle 自动挂载（换个会话入口常常就好了）
4. 偶尔是后端短时故障，等 10–30 分钟再试

### 挂载失败有两种形态，pipeline 都能识别

| 形态 | 预检输出 |
|---|---|
| 完全没有该数据集目录 | `LOL 数据集没找到` / `ExDark 数据集没找到` |
| 目录建出来了但**里面是空的** | `ExDark 挂载不完整（图像 0 张 / 标注 0 个）` |

### 不会再白跑 3 小时

`kaggle_pipeline.py` 现在会**先跑预检**（约 2 分钟，不训练）。
任何一项不过就立刻中止并打印修复建议，日志长这样：

```
[08:00:01 +  0.0min] 步骤 0/3：环境预检（约 2 分钟，不训练）
  [OK  ] LOL 数据集: our485 485 对 / eval15 15 对
  [FAIL] ExDark 挂载不完整（图像 0 张 / 标注 0 个）
[08:00:04 +  0.1min] 【结束】kaggle_preflight.py  退出码 1  用时 0.0 分钟
[!] 预检未通过 —— 已停止，避免白训 3 小时。
```

想跳过预检（不建议）设 `SKIP_PREFLIGHT=1`。

---

## 十二、运行期间怎么看状态

`Save & Run All` 的 Logs 标签**不流式输出 stdout**，所以过程中看不到逐轮日志 —— 这是 Kaggle 的行为，
不是代码没打印。现在 pipeline 的每一段都带**时间戳 + 累计耗时**，跑完后回看日志就能读出时间线：

```
[08:00:22 +  0.1min] 预检通过
[08:00:22 +  0.1min] 消融权重: 已有 无 | 待补 ['baseline_unet', 'plus_illum', 'full_indnet']
[08:00:22 +  0.1min] 步骤 1/3：补齐消融权重（GPU 上约 2.5 – 3.5 小时）
[11:35:10 +215.4min] 【结束】kaggle_ablation_cell.py  退出码 0  用时 214.8 分钟
[11:35:10 +215.4min] 步骤 2/3：ExDark 检测评测（约 15 – 25 分钟）
[11:52:31 +232.1min] 【结束】kaggle_exdark_detect_cell.py  退出码 0  用时 17.3 分钟
```

同时子进程一律以 `-u` 启动（**无缓冲**），避免日志因缓冲而丢失。

**想真正实时看进度**，只能在交互会话里手动运行格子（输出逐行实时出现）。
但交互会话被回收会清空 `/kaggle/working`，权重跟着丢，所以正式跑仍然建议 `Save & Run All`。

---

## 十三、显存不足（CUDA out of memory）与权重找回

### 现象

训练正常结束、三个 `best.pth` 都生成了，但**检测环节**在跑增强时报：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.34 GiB.
GPU 0 has a total capacity of 14.56 GiB of which 3.31 GiB is free.
```

### 根因：ExDark 里混有超大图

ExDark 的图像尺寸不统一，CSV 的 `image_width/height` 分布里出现过 **2560×1920**，
坐标范围也到过 **4301**。而增强网络是 FP32 全分辨率推理：

```
3226×4301 的图，base_channels=32 时仅第一层特征就要 1.65 GiB
→ 解码器拼接处峰值翻倍 → 触发 3.34 GiB 的分配失败
```

### 修复（已内置，无需操作）

| 措施 | 说明 |
|---|---|
| **增强分辨率上限** | `ENH_MAX_SIDE`（默认 1024）。超过就等比缩小 → 增强 → 插值还原到**原尺寸**，保证四种输入分辨率一致、比较公平 |
| **缓解显存碎片** | 在 `import torch` 之前设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| **降档重试** | 单张图仍 OOM 时，自动把上限砍半再试一次 |
| **跳过而不中断** | 极端情况仍失败则跳过该图（计数并在结果里说明），**不让整轮评测白费** |
| 周期释放 | 每 200 张 `empty_cache()` 一次 |

日志里会如实打印：

```
图像最大边长 4301px；增强分辨率上限 1024px
显存处理记录: 缩放后增强 6 次 | 降档重试 0 次 | 跳过 0 张
```

> 检测器本身也会把图缩到 640，所以这一步不会影响"原图 vs 增强"的公平比较。

### 别再重训：从上一个版本的 Output 找回权重

`Save & Run All` 的产物（含 `ablations/*/best.pth`）**会随版本保存**。
下次会话重启后不必重训，按下面做即可：

1. 打开**上一次运行完的那个版本**，在 `Output` 标签确认里面有 `ablations/`
2. 新建 Notebook → 右侧 `Input` → `Add Input` → **`Your Work`** 标签 → 
   找到那个 Notebook / 它的 Output，挂载
3. 运行 pipeline —— 它会自动打印：

```
[11:39:56 +  0.1min] 已从 Input 恢复消融权重: ['baseline_unet', 'plus_illum', 'full_indnet']（省去重新训练）
[11:39:56 +  0.1min] 步骤 1/3：权重已齐全，跳过训练
```

**这样一次运行只要 20–30 分钟**（预检 2 分钟 + 检测 15–25 分钟），而不是 3 小时。

#### 方法 A（推荐）：Add Input → Your Work

1. 打开那个跑成功的 Notebook —— **只浏览，不要点 `Edit`**
2. 切到 **`Output`** 标签，确认里面能看到 `ablations/` 目录（这就是三个权重所在）
3. 新建 Notebook（或打开你要跑的那个）→ 右侧 `Input` → **`Add Input`**
4. 面板里切到 **`Your Work`** 标签 → 找到那个 Notebook（名字形如 `notebook8fcb9429e5`）
   → 点右侧 `+` 添加
5. 挂载后路径形如 `/kaggle/input/<notebook-slug>/ablations/...`

> pipeline 是按 `best.pth` 的位置**递归搜索**的，所以挂在哪一层都能找到，不用管确切路径。

#### 方法 B：把 Output 发布成数据集（方法 A 里找不到 Notebook 时）

1. 打开 Notebook 的 `Output` 标签
2. 用右上角的 `...` 菜单 → **`New Dataset`**（或页面上的 `Create dataset` 按钮）
3. 发布为 **Private** 数据集
4. 回到 Notebook，用 `Add Input` 搜你刚建的那个数据集名

#### 怎么确认成功

pipeline 会打印查找过程（新增）：

```
[hh:mm:ss + 0.0min]   [权重恢复] Input 里共有 3 个 best.pth，前几个位置：
      /kaggle/input/notebook8fcb9429e5/ablations/baseline_unet/best.pth
      /kaggle/input/notebook8fcb9429e5/ablations/plus_illum/best.pth
      /kaggle/input/notebook8fcb9429e5/ablations/full_indnet/best.pth
[hh:mm:ss + 0.0min]   [权重恢复] baseline_unet <- /kaggle/input/...
[hh:mm:ss + 0.1min] 已从 Input 恢复消融权重: ['baseline_unet', 'plus_illum', 'full_indnet']（省去重新训练）
[hh:mm:ss + 0.1min] 步骤 1/3：权重已齐全，跳过训练
```

如果看到 `（一个都没有 —— 说明上次那个版本的 Output 没挂上）`，
说明没有挂上或挂错了版本 —— 换方法 B 再试。

> 注意：把 Output 挂回来后，`/kaggle/input` 里会同时存在**两份代码**
> （数据集里的新版 + Output 里的旧版）。pipeline 会**优先用数据集那份**
> （工作区副本路径含大写 `INDNet`，数据集名为 `indnet-code`，据此区分），
> 不会误跑到没有修复的旧脚本。

---

## 十四、检测评测的两个关键修正

### 1. 通道顺序：numpy 输入必须是 BGR（真 bug，已修）

ultralytics 官方文档明确写：

| 输入类型 | 期望格式 |
|---|---|
| PIL.Image | HWC，**RGB** |
| numpy.ndarray | HWC，**BGR** |
| torch.Tensor | BCHW，RGB |

我们一直用 `Image.open(...).convert("RGB")` 得到 numpy 再直接送进去，
**R/B 通道是互换的**，检测器读到的是错误颜色。所有历史数字（含原图 0.5228）都受此影响。

修正：送检测前统一转换

```python
img_bgr = np.ascontiguousarray(imgs[mode][..., ::-1])
res = detector.predict(img_bgr, conf=0.01, iou=0.5, ...)
```

### 2. 新增两行经典增强基线（Gamma / CLAHE）

这两个模式**与我们自己的模型无关**，但它们回答一个决定性问题：

> **增强这件事本身，对这个检测器有没有好处？**

判读规则（脚本会自动打印结论）：

| 结果 | 含义 | 下一步 |
|---|---|---|
| Gamma/CLAHE **>** 原图 | 增强本身有益，差距来自我们模型的增强质量 | 去修模型（加长训练） |
| Gamma/CLAHE **≤** 原图 | 该检测器在 ExDark 上更偏好原始暗图 | "面向检测"的叙事要重新设计 |

最终表格变为 **6 行**：

```
输入来源                          mAP@0.5
原图（未增强）                       ......
Gamma γ=0.5（经典）                  ......
CLAHE（经典）                        ......
＋残差 U-Net 基线                    ......
＋光照分支                           ......
＋INDNet（完整）                     ......
```

> 用 `EXDARK_CLASSICAL=0` 可关掉这两行（不推荐，它们是最重要的对照）。

---

## 十五、1 分钟诊断：增强图到底长什么样

`notebooks/kaggle_sample_check_cell.py` —— **不训练、不评测**，只处理 12 张图。

做两件事：

1. 导出并排对比图 → `/kaggle/working/detection/samples/cmp_XX_*.png`
   每张 6 格：`原图 | Gamma | CLAHE | 残差U-Net | ＋光照 | INDNet(完整)`
2. 打印统计量（全样本平均）：

| 指标 | 说明 | 危险信号 |
|---|---|---|
| 平均亮度 | 是否真的提亮了 | 远高于原图 = 过度提亮 |
| **高光饱和%** | 亮度 > 250 的像素占比 | **> 10% = 过曝，检测器最怕这个** |
| 暗部压死% | 亮度 < 5 的占比 | 高 = 暗区细节没了 |
| 色彩偏移 R−B | 色偏程度 | 明显偏离原图 = 严重色偏 |
| 亮度 std | 对比度 | 明显变小 = 对比度被抹平 |

> ⚠️ **一个必须避开的度量陷阱：绝对 R−B 不能横向比较。**
> 实测我们的模型输出 `R−B = 26.8`，原图是 `12.5`，看着像"偏色翻倍"——
> 但那只是因为亮度被提亮了 2.6 倍，**绝对色差自然会同比例放大**。
> 按亮度归一后 `(R−B)/Y`：原图 27.9%、Gamma 19.0%、CLAHE 21.2%、
> 残差U-Net 24.5%、＋光照 18.7%、INDNet 完整 21.1% —— **全在同一量级，没有系统性偏色**。
> 所以表格里输出的是**相对色偏%**。诊断格已按此口径修正。

用环境变量 `SAMPLE_N` 可改抽样张数（默认 12）。

**为什么要它**：检测表显示"增强反而伤检测"。要判断是**增强图本身坏了**，
还是**冻结的检测器就是不吃增强图**，最直接的办法就是看一眼 + 看统计。

> 注意：如果 `/kaggle/working` 已被清空，本格会**自动从挂载的 Input 里找回权重**，
> 不需要重新训练。

---

## 十六、`EXDARK_POST_ALIGN`：校色偏对照

`EXDARK_POST_ALIGN=1` 时，每个学习式增强会额外生成一个 `{name}_align` 模式：
**保持亮度不变，只把相对色偏对齐到原图**。用于检验"色偏是否是掉点主因"，无需重训。

配合 `EXDARK_MODES` 只跑关心的模式可大幅省时，例如：

```python
os.environ["EXDARK_POST_ALIGN"] = "1"
os.environ["EXDARK_MODES"] = "original,clahe,full_indnet,full_indnet_align"
os.environ["EXDARK_LIMIT"] = "1500"
```

> 结论提示：由于各方法的**相对**色偏本来就在同一量级（19%~28%），
> 这个对照很可能显示"校色偏并不能救回 mAP" —— 那反而能把矛头
> 更明确地指向模型质量本身（PSNR 18.3 vs SOTA 25.2）。

---

## 十七、检测器微调评测（协议 A2）—— 让增强获得域适配的机会

### 为什么要换协议

冻结协议（一个 COCO 预训练检测器直接测原图 / 增强图）的实测结果是负的：

```
原图 0.5338 | Gamma 0.4981 | CLAHE 0.5348(+0.001) | 残差U-Net -0.079 | +光照 -0.154 | INDNet -0.063
```

**连 CLAHE 都只拿到 +0.001**（mAP 在 7362 张图上的采样噪声约 ±0.005）。这说明问题不在
"我们的模型不够好"，而在**协议本身**：COCO 预训练检测器在 ExDark 上存在域偏移，
增强又会进一步改变图像分布，两头不利。文献里报增益的工作基本都做了域适配或检测感知训练。

### 协议怎么设计的

**每条 pipeline 各自微调一个检测器**，在自己的增强训练集上训练、在自己的增强测试集上测试：

| 输入来源 | 各自微调的检测器 | 训练数据 | 测试数据 |
|---|---|---|---|
| 原图 | detector_original | 原图 train | 原图 test |
| Gamma γ=0.5 | detector_gamma | Gamma train | Gamma test |
| CLAHE | detector_clahe | CLAHE train | CLAHE test |
| ＋残差 U-Net | detector_baseline | 该变体 train | 该变体 test |
| ＋光照分支 | detector_illum | 该变体 train | 该变体 test |
| ＋INDNet（完整） | detector_full | 该变体 train | 该变体 test |

→ 每条 pipeline 都是域内，谁都不吃亏；差别只剩"哪种前处理更有利于检测"。

### 钉死的公平性（写在代码里，不允许逐方法调参）

| 约束 | 做法 |
|---|---|
| 同一份划分 | 按**图像**切分（60/15/25），seed=3407，写入 `exdark_split.json`，所有 pipeline 共用 |
| 相同超参 | epochs / lr / imgsz / batch / 优化器 / 随机种子全部固定 |
| 训练预算相同 | `patience >= epochs`，**关闭早停** |
| 检测器不偷看 | 每个检测器的训练集里只有自己那种图像 |
| 关闭干扰增强 | `hsv_h/s/v=0`、`mosaic=0`、`mixup=0`、`copy_paste=0` —— 否则检测器自带的颜色增强会掩盖"不同增强方法"的差异，而那恰恰是被测对象 |
| 尺寸路径一致 | 所有图像先等比缩到长边 ≤ `DATASET_MAX_SIDE`（默认 1024）**再做增强**，同一条尺寸路径 |

### 输出两张表

**表 A（主表）：域内微调协议** —— 公平比较"哪种增强最有利于检测"。

**表 B：交叉评测** —— 用 detector_original 去测其它 pipeline 的测试图，
对应真实场景（已有检测器，只是换个前处理）。两张表一起看，才能分清
"增强本身的效果"与"域适配带来的收益"各自有多大。

### 怎么跑

**方式一：并入一键链路（推荐）**

```python
os.environ["RUN_FINETUNE"] = "1"
os.environ["FT_MODES"] = "original,clahe,full_indnet"   # 先跑子集看清趋势
```

然后粘 `notebooks/kaggle_pipeline.py`，`Save & Run All`。

**方式二：单独一格**

新建一格，粘 `notebooks/kaggle_exdark_finetune_cell.py` 全部内容。

### 参数速查

| 变量 | 默认 | 作用 |
|---|---|---|
| `FT_EPOCHS` | 40 | 每个检测器的微调轮数 |
| `FT_MODES` | 全部 6 个 | 只跑指定模式（逗号分隔），省时间 |
| `FT_LIMIT` | 0（全部） | 每类只取前 N 张，冒烟验证用 |
| `FT_BATCH` / `FT_IMGSZ` | 16 / 640 | 训练批大小与输入尺寸 |
| `FT_SEED` | 3407 | 划分与训练的随机种子 |
| `FT_CROSS_EVAL` | 1 | 是否额外输出交叉矩阵（表 B） |
| `DATASET_MAX_SIDE` | 1024 | 图像长边上限（训练与评测统一） |
| `FT_CLEAN_TRAIN` | 0 | 每个模式训完即删训练图，省磁盘 |
| `FT_FORCE` | 0 | 强制重做图像与重训 |
| `RUN_FINETUNE` | 0 | 在 pipeline 里追加这一步 |

### 耗时与磁盘（GPU T4）

| 阶段 | 耗时 |
|---|---|
| 图像准备（6 模式 × 7362 张） | 约 12 – 15 分钟 |
| 微调（6 个检测器 × 40 轮） | 约 80 – 100 分钟 |
| 评测（表 A 6 次 + 表 B 5 次） | 约 15 分钟 |
| **合计** | **约 2 小时** |

磁盘：6 个模式共约 4 – 6 GB（`/kaggle/working` 若紧张，用 `FT_CLEAN_TRAIN=1`
或 `FT_MODES` 只跑子集）。

### 怎么读结果

| 表 A 的结果 | 含义 | 下一步 |
|---|---|---|
| 某个增强 > 原图且差距 > 0.005 | 域适配后增强确实有帮助 | 可支撑 `Detection-Oriented` 叙事，进论文主表 |
| 全部与原图持平（±0.005 内） | 增强对检测中性 | 叙事要调整，比如主打解耦本身 |
| 全部低于原图 | 域适配也救不回来 | 「面向检测」这条路要重新设计 |

> 表 B 若明显低于表 A，说明"换前处理"会引入域偏移、实际部署需要重新微调检测器 ——
> 这本身就是值得写进论文的发现，不要丢掉。

### 幂等性

- 划分写入 `exdark_split.json`，重跑自动复用
- 已存在微调权重则跳过训练
- 数据集目录会**自动清理过期文件**（换了 `FT_LIMIT` 或划分之后，旧图不会被残留污染）
- 每个模式打印实际文件数与划分数的对比，不符会标出 `[!]`

---

## 十八、训练预算的重大修正（2026-10-16）—— 为什么只有 18.3 dB

### 发现

对照 Retinexformer 论文的 Implementation Details：

> *"The model is trained with the Adam optimizer (β₁=0.9, β₂=0.999) for
> **2.5 × 10⁵ iterations**. The learning rate is initially set to 2×10⁻⁴ and then
> steadily decreased to 1×10⁻⁶ by the cosine annealing scheme. Patches at the size of
> **128 × 128** are randomly cropped... The **batch size is 8**.
> ... augmented with random rotation and flipping. The training objective is to minimize
> the **MAE** between the lit-up image and enhanced image."*

我们的旧配方：`epochs: 200` × (485/16 = 30 iter) = **6,000 次迭代**。

| | Retinexformer | 我们（旧） | 比值 |
|---|---|---|---|
| 迭代数 | 250,000 | 6,000 | **2.4%** |
| patch | 128×128 | 256×256 | 4× |
| batch | 8 | 16 | 2× |
| **总像素预算** | 3.28×10¹⁰ | 6.29×10⁹ | **19.2%** |

**我们只跑了参考配方的 2.4% 迭代数、19% 的像素预算。**
这就是 18.3 dB vs 25.16 dB 最可能的主因 —— 而不是模型结构。

### 修正内容

**1. 训练脚本支持按「总迭代数」训练**

`configs/*.yaml` 里新增 `train.total_iter`：

```yaml
train:
  total_iter: 60000     # >0 时按迭代数训练，epochs 会被自动换算
  epochs: 200           # total_iter=0 时才生效（保留旧的按 epoch 方式）
```

换算：`patch 128 / batch 8 → 485//8 = 60 iter/epoch`
- `60000 iter` ≈ 1000 epoch ≈ **2 小时**（T4）
- `250000 iter` ≈ 4167 epoch ≈ **5–7 小时**（T4）

余弦调度的周期会跟随实际训练长度（`T_max` 用换算后的 epoch 数），不会出现
"lr 早就降到 0 却还在训"的问题。

**2. 配置对齐参考配方**

| 项 | 旧 | 新 | 依据 |
|---|---|---|---|
| `patch_size` | 256 | **128** | 与 Retinexformer 一致 |
| `batch_size` | 16 | **8** | 同上 |
| `augment.rot90` | false | **true** | "random rotation and flipping" |
| `w_perceptual` | 0.01 | **0.0** | 参考配方只用 MAE；感知损失会略降 PSNR，还依赖联网下 VGG |

**3. 修掉一个真 bug：几何增强开关被忽略**

`datasets/lol_dataset.py` 里 `_geometric_augment` 原来是 `@staticmethod`，
**写死 hflip + vflip**，配置里的 `rot90` 和 `vflip` 完全不生效
（`build_datasets` 里那段 "hflip=false 时整体替换" 的 hack 还会顺手把 vflip 也关掉）。

现已改为实例方法，`hflip / vflip / rot90` **三个开关逐个独立生效**，
且 low / high 严格同步变换。

**4. 消融格加了「配方指纹」守卫**

旧配方训出的 `best.pth` 就在同一个目录里。如果只判断"文件存在"就跳过训练，
会拿旧配方的结果冒充新实验。现在会比对
`(total_iter, epochs, patch_size, batch_size)`：

```
  已有权重但训练配方不同，归档后重训
    旧配方 total_iter/epochs/patch/batch = (0, 200, 256, 16)
    新配方 total_iter/epochs/patch/batch = (60000, 200, 128, 8)
```
旧权重会被改名为 `best_old_recipe.pth` 而不是删除。

### 建议的分阶段执行

**阶段 1：探测（约 2 小时）**

```python
os.environ["TOTAL_ITER"]   = "60000"
os.environ["ABL_VARIANTS"] = "full_indnet"    # 只训完整模型，省钱
```

只看一件事：**PSNR 从 18.3 涨到多少**。

| 60k iter 后的 PSNR | 结论 | 下一步 |
|---|---|---|
| ≥ 21 | 训练量确实是主因，还有很大空间 | 跑满 250k（约 6 小时）× 3 变体 |
| 19–21 | 部分有效，但结构也是瓶颈 | 跑满 250k + 考虑提升模型容量 |
| ≤ 19 | 训练量不是主因，是架构问题 | 停止加训练量，换 backbone |

**阶段 2：正式（视阶段 1 结果）**

```python
os.environ["TOTAL_ITER"] = "250000"
# ABL_VARIANTS 留空 = 三个变体全跑（约 15–20 小时，需要跨多次会话）
```

### 环境变量速查（消融格）

| 变量 | 默认 | 作用 |
|---|---|---|
| `TOTAL_ITER` | 用配置值 | 覆盖总迭代数 |
| `PATCH_SIZE` | 用配置值 | 覆盖裁剪尺寸 |
| `BATCH_SIZE` | 用配置值 | 覆盖批大小 |
| `ABL_VARIANTS` | 全部 3 个 | 逗号分隔，只跑指定变体 |
| `RETRAIN` | 0 | 强制删除旧结果重训 |

---

## 十九、★ 陷阱：改了训练配方，却"跑了两小时还是老数字"（2026-10-17 实际踩到）

### 现象

设好 `TOTAL_ITER=60000`、`ABL_VARIANTS=full_indnet` 后 `Save & Run All`，
**`Successfully ran in 8824.1s`（2 小时 27 分）看起来一切正常**，产物清单也齐全。
但一查：

```
ablations/full_indnet/ 里只有 best.pth
```

**没有 `train.log`、没有 `config.yaml`、没有中间快照** —— 也就是说**根本没有训练**，
那 2.5 小时花在检测和 A2 微调上，而检测/A2 用的还是**旧的 18.3 dB 权重**。

### 根因（pipeline 的逻辑缺陷）

`kaggle_pipeline.py` 里两段逻辑的顺序有问题：

```python
restored = restore_weights_from_input()      # ← ① 先把旧版本的权重复制进 ablations/
have = [n for n in REQUIRED if (ABL/n/"best.pth").is_file()]
todo = [n for n in REQUIRED if n not in have]
if todo:   run_ablation_cell()               # ← ② 全都"已存在" → 永远走不到这里
else:      print("权重已齐全，跳过训练")
```

只要**挂载了旧版本的 Output**（为了省权重下载时间），①就会把三个 `best.pth` 放到位，
于是 ②判定"无事可做"直接跳过。**而配方指纹守卫写在消融格内部 —— 消融格根本没被调用。**

**这是个"静默失效"：没有报错、没有警告、耗时正常、产物齐全，只是数字是旧的。**

### 修好的做法

| 位置 | 改动 |
|---|---|
| `kaggle_pipeline.py` | **总是调用消融格**，由它内部的配方指纹决定训练还是跳过（无事可做时只需 2 秒）。新增 `SKIP_ABLATION=1` 逃生开关 |
| `kaggle_pipeline.py` | 恢复权重后**打印每个权重的配方指纹**，并在"步骤 1"里打印本次 `TOTAL_ITER` / `ABL_VARIANTS` |
| `kaggle_ablation_cell.py` | 开跑前打印**训练计划**，逐个变体标注 `[训练]` / `[跳过]` 及原因 |

修好后的日志长这样（一眼就能看出会不会真训）：

```
[hh:mm:ss + 0.0min]   baseline_unet 现有权重配方 (total_iter, epochs, patch, batch) = (0, 200, 256, 16)
[hh:mm:ss + 0.0min]   full_indnet 现有权重配方 (total_iter, epochs, patch, batch) = (0, 200, 256, 16)
[hh:mm:ss + 0.1min] 步骤 1/3：消融/训练（配方指纹决定是否真训；TOTAL_ITER=60000 ABL_VARIANTS=full_indnet）
======================================================================
消融 / 训练计划
======================================================================
  本次配方 (total_iter, epochs, patch, batch) = (60000, 200, 128, 8)
  [训练] INDNet (完整)        配方不同：(0, 200, 256, 16) != (60000, 200, 128, 8)
======================================================================
```

### 怎么自查（三条，任何一条都足够）

1. **看 `ablations/<变体>/` 里有没有 `config.yaml` 和 `train.log`** ——
   只有 `best.pth` 就说明是"从 Input 恢复进来的"，没训练过
2. **看产物时间** —— 训练产出的 `best.pth` 应该是本次运行时间
3. **看日志里有没有 `训练计划` 那一段** —— 没有就说明消融格压根没跑

### 想强制重训的三种办法（任选）

| 办法 | 说明 |
|---|---|
| 不挂旧版本 Output | 最干净：没有可恢复的权重，必然训练 |
| `RETRAIN=1` | 消融格会先 `rmtree` 三个变体目录再训 |
| 靠配方指纹 | 更新代码包后自动生效：检测到配方不同会归档旧权重并重训 |

> ⚠️ 注意 `RETRAIN=1` 会删掉**全部三个**变体目录，而 `ABL_VARIANTS` 只控制训练哪几个。
> 探测阶段用「不挂旧 Output」最省事。

### 顺带：Output 体积

那次 2.27 GB 主要是 A2 留下的 `yolo_ds/`（3 个模式 × 7362 张 JPEG）。
**探测阶段不要把 `RUN_FINETUNE` 打开**，产物只有几十 MB。
另外 `save_every` 已从 30 调到 200（每个快照约 15 MB，早期 33 个快照会白占 0.5 GB）。
