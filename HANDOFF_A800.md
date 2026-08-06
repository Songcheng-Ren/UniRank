# UniRank A800 Handoff

## 目标

在 A800 开发机上下载预处理数据，并使用 UniRank 的真实训练入口完成一次可运行的单机多卡实验。

本地 Mac 已完成以下验证：

- Python/PyTorch/项目模块可以正常导入。
- 配置可以解析。
- 特征处理器、`feature_map`、模型、优化器和 `torch.compile` 可以初始化。
- `run_expid.py` 已进入真实数据加载阶段。
- 本机没有完整 Parquet 数据，因此没有完成训练。

仓库中已有两个与跨机器运行相关的修复：

- `unirank/utils.py` 支持通过 `UNIRANK_DATA_ROOT` 重定位配置中的集群数据路径。
- `run_expid.py` 修复 `--enable_bf16 false` 的布尔参数解析。

不要回退这两个修改。

---

## 推荐执行顺序

### 1. 进入仓库并检查 GPU

```bash
cd /path/to/UniRank

nvidia-smi
python3 --version
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

如果 `torch.cuda.is_available()` 为 `False`，不要开始训练，先安装匹配 A800 驱动的 CUDA PyTorch。

A800 通常使用 8 张 GPU；实际 GPU 数量以 `nvidia-smi` 为准。

---

### 2. 创建 Python 环境并安装依赖

推荐 Python 3.10，和仓库 README 的环境保持一致：

```bash
conda create -n UniRank python=3.10 -y
conda activate UniRank
```

安装与驱动匹配的 CUDA 版本 PyTorch。README 中的示例是 CUDA 12.6：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \\
  --index-url https://download.pytorch.org/whl/cu126
```

如果服务器驱动不支持 CUDA 12.6，需要根据服务器实际驱动选择兼容的 PyTorch CUDA wheel，不要盲目使用上面的命令。

安装项目依赖：

```bash
pip install -r requirements.txt
```

验证依赖：

```bash
python3 - <<'PY'
import torch
import pandas
import pyarrow
import polars
import yaml
import heavyball

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("pandas:", pandas.__version__)
print("pyarrow:", pyarrow.__version__)
print("polars:", polars.__version__)
print("dependencies: OK")
PY
```

如果 `heavyball` 在 Python 3.9 报类型注解兼容错误，优先使用 Python 3.10，不要在 A800 上继续使用系统 Python 3.9。

---

### 3. 安装 Hugging Face CLI

```bash
pip install -U huggingface_hub
hf --help
```

如果 Hugging Face 需要鉴权：

```bash
hf auth login
```

使用个人 access token，不要把 token 写入仓库、命令脚本或日志。

---

## 数据集选择

### 最小数据集：Taobao

UniRank README 中公布的预处理后样本量约为：

| 数据集 | 样本量 | UniRank dataset ID | Hugging Face dataset |
|---|---:|---|---|
| Taobao | 2,360 万 | `Taobao_Action` | `salmon1802/Taobao` |
| MerRec | 1.72 亿 | `MerRec_Action` | `salmon1802/MerRec` |
| KuaiRand | 3.23 亿 | `KuaiRand_Video_Action` | `salmon1802/KuaiRand` |
| QK-Video | 4.93 亿 | `QK_Video_Action` | `salmon1802/QK-Video` |
| TAAC-25 | 7.57 亿 | `TencentGR_10M_Action` | `salmon1802/TAAC-25` |

第一次在 A800 上验证，推荐使用 **Taobao**，因为它最小，下载和首次运行成本较低。

如果任务明确要求复现 KuaiRand，再下载 KuaiRand；不要一开始下载所有数据集。

---

## 数据下载

### 方案 A：下载 Taobao（推荐首次 smoke run）

在仓库根目录执行：

```bash
mkdir -p data/Taobao_Action

hf download salmon1802/Taobao \\
  --repo-type dataset \\
  --local-dir "$PWD/data/Taobao_Action"
```

设置数据根目录：

```bash
export UNIRANK_DATA_ROOT="$PWD/data"
```

### 方案 B：下载 KuaiRand

```bash
mkdir -p data/KuaiRand_Video_Action

hf download salmon1802/KuaiRand \\
  --repo-type dataset \\
  --local-dir "$PWD/data/KuaiRand_Video_Action"

export UNIRANK_DATA_ROOT="$PWD/data"
```

### 其他数据集

```bash
# QK-Video
mkdir -p data/QK_Video_Action
hf download salmon1802/QK-Video \\
  --repo-type dataset \\
  --local-dir "$PWD/data/QK_Video_Action"

# TAAC-25
mkdir -p data/TencentGR_10M_Action
hf download salmon1802/TAAC-25 \\
  --repo-type dataset \\
  --local-dir "$PWD/data/TencentGR_10M_Action"

# MerRec
mkdir -p data/MerRec_Action
hf download salmon1802/MerRec \\
  --repo-type dataset \\
  --local-dir "$PWD/data/MerRec_Action"
```

每个数据集必须下载到独立目录，目录名要和配置中的 `dataset_id` 一致。

---

## 下载后必须检查目录

以 Taobao 为例：

```bash
find data/Taobao_Action -maxdepth 3 -type f | head -50
```

必须存在以下三组目录：

```text
data/Taobao_Action/train/data
data/Taobao_Action/train/user_info
data/Taobao_Action/train/item_info

data/Taobao_Action/valid/data
data/Taobao_Action/valid/user_info
data/Taobao_Action/valid/item_info

data/Taobao_Action/test/data
data/Taobao_Action/test/user_info
data/Taobao_Action/test/item_info
```

检查 Parquet 文件：

```bash
find data/Taobao_Action -type f -name '*.parquet' | head
find data/Taobao_Action -type f -name '*.parquet' | wc -l
```

检查每个 split 是否有匹配的 block 文件：

```bash
find data/Taobao_Action/train -type f -name 'part-*.parquet' | sort | head
find data/Taobao_Action/valid -type f -name 'part-*.parquet' | sort | head
find data/Taobao_Action/test -type f -name 'part-*.parquet' | sort | head
```

检查元数据：

```bash
find data/Taobao_Action -name meta_data.json -o -name block_manifest.json
```

如果下载工具把内容多包了一层，例如：

```text
data/Taobao_Action/Taobao/...
```

不要直接运行。先把真正包含 `train/valid/test` 的目录调整到：

```text
data/Taobao_Action/
```

---

## 第一次运行：单进程 GPU smoke run

先用 Taobao 配置验证数据加载、模型构建和至少一个训练 batch：

```bash
export UNIRANK_DATA_ROOT="$PWD/data"

python3 run_expid.py \\
  --config ./config \\
  --expid RankMixer_Taobao_Action \\
  --gpu 0 \\
  --enable_bf16 true
```

注意：当前实验配置可能会跑完整的一个 epoch。第一次验证时，如果只想快速确认入口，可以让 agent 先观察：

- 是否正确加载 `feature_map.json`；
- 是否完成模型和优化器初始化；
- 是否成功读取第一个 Parquet block；
- 是否出现 loss、step 或 validation 日志。

不要因为运行时间较长就同时启动第二个训练进程。

如果需要缩短首次 GPU 验证时间，应优先使用较小的实验配置或临时降低配置中的 `batch_size/max_len/num_workers`，并明确记录这些改动；不要直接修改默认配置后提交而不说明。

---

## 正式运行：单机 8 卡 DDP

确认单卡 smoke run 可以加载数据后，再启动多卡：

```bash
export UNIRANK_DATA_ROOT="$PWD/data"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TMPROOT="$(mktemp -d /tmp/unirank.XXXXXX)"
mkdir -p "$TMPROOT/torchinductor" "$TMPROOT/triton" "$TMPROOT/torch_extensions"

TMPDIR="$TMPROOT" \\
TMP="$TMPROOT" \\
TEMP="$TMPROOT" \\
TORCHINDUCTOR_CACHE_DIR="$TMPROOT/torchinductor" \\
TRITON_CACHE_DIR="$TMPROOT/triton" \\
TORCH_EXTENSIONS_DIR="$TMPROOT/torch_extensions" \\
torchrun \\
  --standalone \\
  --master_port=29500 \\
  --nproc_per_node=8 \\
  run_expid.py \\
  --config ./config \\
  --expid RankMixer_Taobao_Action \\
  --gpu 0,1,2,3,4,5,6,7 \\
  --enable_bf16 true
```

如果只有 4 张 GPU，将所有相关参数同步改为：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun 参数
--nproc_per_node=4

# run_expid.py 参数
--gpu 0,1,2,3
```

不要出现以下不一致：

```text
--nproc_per_node=8 但 --gpu 只有 4 张卡
--nproc_per_node=4 但 --gpu 写了 8 个 ID
```

---

## 使用 run_all.sh 时的注意事项

`run_all.sh` 默认设置为 4 GPU，并且当前脚本中有一个启用的实验：

```bash
run_exp "OneTrans_KuaiRand_Video_Action_Ablation_NumLayers_Large"
```

不要直接执行未检查的 `./run_all.sh`，因为它可能：

- 使用 KuaiRand 而不是已下载的 Taobao；
- 启动 4 卡 DDP；
- 跑较大的 ablation；
- 使用与当前 A800 GPU 数量不匹配的设置。

首次运行应优先直接调用 `run_expid.py`，明确指定 `--expid`。如果必须使用 `run_all.sh`，先检查并修改：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8
```

同时只保留一个确认过的数据集实验，并把其他 `run_exp` 行注释掉。

---

## 推荐实验命令

### Taobao smoke run

```bash
export UNIRANK_DATA_ROOT="$PWD/data"
python3 run_expid.py \\
  --gpu 0 \\
  --expid RankMixer_Taobao_Action \\
  --enable_bf16 true
```

### Taobao 8 卡正式运行

```bash
export UNIRANK_DATA_ROOT="$PWD/data"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --master_port=29500 --nproc_per_node=8 \\
  run_expid.py \\
  --config ./config \\
  --expid RankMixer_Taobao_Action \\
  --gpu 0,1,2,3,4,5,6,7 \\
  --enable_bf16 true
```

### KuaiRand 8 卡正式运行

确认已下载 KuaiRand 后：

```bash
export UNIRANK_DATA_ROOT="$PWD/data"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --master_port=29500 --nproc_per_node=8 \\
  run_expid.py \\
  --config ./config \\
  --expid RankMixer_KuaiRand_Video_Action \\
  --gpu 0,1,2,3,4,5,6,7 \\
  --enable_bf16 true
```

---

## 日志、checkpoint 和结果

默认 checkpoint 根目录是：

```text
./checkpoints/
```

日志通常位于对应 checkpoint/data set 目录或 `logs/`，具体以启动输出为准。

运行时需要记录：

- Git commit hash；
- 数据集名称和 Hugging Face revision；
- `UNIRANK_DATA_ROOT`；
- GPU 数量和 GPU 型号；
- `max_len`；
- `batch_size`；
- `token_dim`；
- 是否开启 BF16；
- `torch` 和 CUDA 版本；
- 最终 checkpoint 和验证指标。

运行前建议保存：

```bash
git rev-parse HEAD
python3 -V
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
nvidia-smi
```

---

## 常见问题

### 1. `FileNotFoundError: Parquet path not found`

检查：

```bash
echo "$UNIRANK_DATA_ROOT"
find "$UNIRANK_DATA_ROOT/Taobao_Action" -type f -name '*.parquet' | head
```

确认数据目录与 `--expid` 一致：

```text
RankMixer_Taobao_Action       -> data/Taobao_Action
RankMixer_KuaiRand_Video_Action -> data/KuaiRand_Video_Action
```

### 2. `No matched blocked parquet part ids`

说明 `data`、`user_info`、`item_info` 三类 block 文件的 `part-xxxxx.parquet` 编号不匹配。确认三类目录来自同一个 Hugging Face 预处理数据集版本，不要混用不同下载版本。

### 3. CUDA 不可用

```bash
python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
nvidia-smi
```

确认当前 shell 使用的是 A800 环境中的 conda Python，而不是系统 Python。

### 4. DDP 端口冲突

换一个端口：

```bash
--master_port=29501
```

### 5. 显存不足

按以下顺序处理：

1. 确认没有其他训练进程占用 GPU；
2. 降低 `batch_size`；
3. 增大 `accumulation_steps` 以保持有效 batch size；
4. 降低 `max_len`；
5. 确认是否误启动了比目标更大的 ablation；
6. 不要首先删除 BF16，因为 A800 适合 BF16。

### 6. `torch.compile` 首次很慢

这是正常现象。首次编译会生成 Inductor/Triton cache，等待编译完成，不要重复启动多个相同任务。

### 7. 磁盘空间不足

数据集、checkpoint、Parquet cache 和编译 cache 都可能占用大量空间：

```bash
df -h
 du -sh data checkpoints /tmp 2>/dev/null || true
```

---

## Agent 完成标准

只有满足以下条件才报告“运行成功”：

1. 数据下载完成且目录结构正确；
2. `feature_map` 能加载；
3. 模型和优化器初始化成功；
4. 至少读取并处理一个真实 Parquet batch；
5. 日志中出现训练 step/loss 或明确的验证结果；
6. 记录实际使用的命令、GPU 数量和数据集路径。

如果只完成模型初始化但在 Parquet 加载前失败，不要报告训练成功；应继续排查数据目录、block 文件和元数据。
