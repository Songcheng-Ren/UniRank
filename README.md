<p align="center">
  <img src="./assets/figures/unirank_logo.png" alt="UniRank logo" width="720">
</p>

# UniRank: Benchmarking Ranking Models for Unified Sequential Modeling and Feature Interaction <sub>[v0.7.1](https://github.com/salmon1802/UniRank/tree/v0.7.1)</sub>

UniRank is an open PyTorch benchmark for unified sequential modeling and feature interaction in large-scale recommendation ranking. It standardizes chronological point-wise autoregressive supervision, multi-feedback evaluation, model implementations, data processing, and distributed training in one reproducible pipeline.

The benchmark contains fifteen unified ranking architectures and five industrial datasets from short-video, advertising, and e-commerce scenarios. Their sequence lengths span roughly `10^2` to `10^5`. The toolkit supports blocked Parquet loading, DDP, operator compilation, mixed precision, optimized attention, and activation checkpointing so that accuracy and efficiency can be compared under the same protocol.

## Why UniRank?

Modern ranking research is moving from isolated sequence pooling and feature-cross modules toward unified architectures that allow behavioral tokens, target items, and non-sequential fields to interact in a shared representation space. Comparing these models is difficult because published systems often use different datasets, split rules, sequence definitions, label semantics, and training infrastructure.

UniRank is designed to make the following questions measurable under a common protocol:

- Which architecture performs best when the data split, features, sequence length, tasks, and metrics are fixed?
- Should sequence modeling happen before feature interaction, or should both happen layer by layer?
- How do model conclusions change across click, engagement, cart, and conversion objectives?
- How do model size, token dimension, and history length affect accuracy, memory, and throughput?
- Which systems techniques are required to train unified rankers on datasets that do not fit in host memory?

The goal is not to hide dataset-specific semantics. UniRank makes those choices explicit in preprocessing scripts and YAML configurations so that a result can be traced from raw events to labels, history tokens, model inputs, and final metrics.

The project makes three main contributions:

- **An open unified-ranking benchmark:** fifteen recent architectures are evaluated on five large industrial datasets under a common chronological, multi-feedback protocol.
- **A practical large-scale toolkit:** DDP, compilation, mixed precision, optimized attention, activation checkpointing, and blocked loading reduce the systems barrier to reproducing large ranking models.
- **A reproducible empirical study:** the repository releases preprocessing code, configurations, model implementations, evaluation code, and benchmark results for analyzing model--data and model--task affinity.

## Models

The following implementations are exported by `model_zoo/__init__.py`:

<table>
  <thead>
    <tr>
      <th width="5%" align="center">No.</th>
      <th width="11%" align="center">Publication</th>
      <th width="13%">Model</th>
      <th width="27%">Affiliation</th>
      <th width="44%">Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1</td>
      <td align="center">arXiv'23</td>
      <td><a href="./model_zoo/HiFormer.py">HiFormer</a></td>
      <td><img src="https://cdn.simpleicons.org/google/4285F4" alt="Google" height="18"> Google</td>
      <td><a href="https://arxiv.org/pdf/2311.05884">HiFormer: Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems</a></td>
    </tr>
    <tr>
      <td align="center">2</td>
      <td align="center">CIKM'25</td>
      <td><a href="./model_zoo/RankMixer.py">RankMixer</a></td>
      <td><img src="https://cdn.simpleicons.org/bytedance/3C8CFF" alt="ByteDance" height="18"> ByteDance</td>
      <td><a href="https://arxiv.org/abs/2507.15551">RankMixer: Scaling Up Ranking Models in Industrial Recommenders</a></td>
    </tr>
    <tr>
      <td align="center">3</td>
      <td align="center">arXiv'25</td>
      <td><a href="./model_zoo/INFNet.py">INFNet</a></td>
      <td><img src="https://cdn.simpleicons.org/kuaishou/FF4906" alt="Kuaishou" height="18"> Kuaishou</td>
      <td><a href="https://arxiv.org/pdf/2508.11565v1">INFNet: A Task-aware Information Flow Network for Large-Scale Recommendation Systems</a></td>
    </tr>
    <tr>
      <td align="center">4</td>
      <td align="center">RecSys'25</td>
      <td><a href="./model_zoo/LONGER.py">LONGER</a></td>
      <td><img src="https://cdn.simpleicons.org/bytedance/3C8CFF" alt="ByteDance" height="18"> ByteDance</td>
      <td><a href="https://arxiv.org/abs/2505.04421">LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders</a></td>
    </tr>
    <tr>
      <td align="center">5</td>
      <td align="center">WWW'26</td>
      <td><a href="./model_zoo/OneTrans.py">OneTrans</a></td>
      <td><img src="https://www.google.com/s2/favicons?domain=ntu.edu.sg&amp;sz=64" alt="NTU" height="18"> NTU<br><img src="https://cdn.simpleicons.org/bytedance/3C8CFF" alt="ByteDance" height="18"> ByteDance</td>
      <td><a href="https://arxiv.org/abs/2510.26104">OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender</a></td>
    </tr>
    <tr>
      <td align="center">6</td>
      <td align="center">arXiv'26</td>
      <td><a href="./model_zoo/Zenith.py">Zenith</a></td>
      <td><img src="https://www.google.com/s2/favicons?domain=ncsu.edu&amp;sz=64" alt="NCSU" height="18"> NCSU<br><img src="https://cdn.simpleicons.org/tiktok/000000/FFFFFF" alt="TikTok" height="18"> TikTok<br><img src="https://cdn.simpleicons.org/bytedance/3C8CFF" alt="ByteDance" height="18"> ByteDance</td>
      <td><a href="https://arxiv.org/pdf/2601.21285">Zenith: Scaling up Ranking Models for Billion-scale Livestreaming Recommendation</a></td>
    </tr>
    <tr>
      <td align="center">7</td>
      <td align="center">SIGIR'26</td>
      <td><a href="./model_zoo/HyFormer.py">HyFormer</a></td>
      <td><img src="https://cdn.simpleicons.org/bytedance/3C8CFF" alt="ByteDance" height="18"> ByteDance</td>
      <td><a href="https://arxiv.org/abs/2601.12681">HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction</a></td>
    </tr>
    <tr>
      <td align="center">8</td>
      <td align="center">KDD'26</td>
      <td><a href="./model_zoo/MixFormer.py">MixFormer</a></td>
      <td><img src="https://cdn.simpleicons.org/bytedance/3C8CFF" alt="ByteDance" height="18"> ByteDance</td>
      <td><a href="https://arxiv.org/abs/2602.14110">MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders</a></td>
    </tr>
    <tr>
      <td align="center">9</td>
      <td align="center">KDD'26</td>
      <td><a href="./model_zoo/TokenMixer.py">TokenMixer</a></td>
      <td><img src="https://cdn.simpleicons.org/bytedance/3C8CFF" alt="ByteDance" height="18"> ByteDance</td>
      <td><a href="https://arxiv.org/pdf/2602.06563">TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders</a></td>
    </tr>
    <tr>
      <td align="center">10</td>
      <td align="center">KDD'26</td>
      <td><a href="./model_zoo/EST.py">EST</a></td>
      <td><img src="https://cdn.simpleicons.org/alibabacloud/FF6A00" alt="Alibaba" height="18"> Alibaba</td>
      <td><a href="https://arxiv.org/pdf/2602.10811">EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling</a></td>
    </tr>
    <tr>
      <td align="center">11</td>
      <td align="center">arXiv'26</td>
      <td><a href="./model_zoo/HeMix.py">HeMix</a></td>
      <td><img src="https://cdn.simpleicons.org/alibabacloud/FF6A00" alt="Alibaba" height="18"> Alibaba</td>
      <td><a href="https://arxiv.org/pdf/2602.09387">Query-Mixed Interest Extraction and Heterogeneous Interaction: A Scalable CTR Model for Industrial Recommender Systems</a></td>
    </tr>
    <tr>
      <td align="center">12</td>
      <td align="center">arXiv'26</td>
      <td><a href="./model_zoo/UniMixer.py">UniMixer</a></td>
      <td><img src="https://cdn.simpleicons.org/kuaishou/FF4906" alt="Kuaishou" height="18"> Kuaishou</td>
      <td><a href="https://arxiv.org/pdf/2604.00590">UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems</a></td>
    </tr>
    <tr>
      <td align="center">13</td>
      <td align="center">arXiv'26</td>
      <td><a href="./model_zoo/TokenFormer.py">TokenFormer</a></td>
      <td><img src="https://www.google.com/s2/favicons?domain=tencent.com&amp;sz=64" alt="Tencent" height="18"> Tencent</td>
      <td><a href="https://arxiv.org/abs/2604.13737">TokenFormer: Unify the Multi-Field and Sequential Recommendation Worlds</a></td>
    </tr>
    <tr>
      <td align="center">14</td>
      <td align="center">arXiv'26</td>
      <td><a href="./model_zoo/UltraHSTU.py">UltraHSTU</a></td>
      <td><img src="https://cdn.simpleicons.org/meta/0866FF" alt="Meta" height="18"> Meta</td>
      <td><a href="https://arxiv.org/pdf/2602.16986">Bending the Scaling Law Curve in Large-Scale Recommendation Systems</a></td>
    </tr>
    <tr>
      <td align="center">15</td>
      <td align="center">SIGIR'26</td>
      <td><a href="./model_zoo/SSR.py">SSR</a></td>
      <td><img src="https://cdn.simpleicons.org/alibabacloud/FF6A00" alt="Alibaba" height="18"> Alibaba</td>
      <td><a href="https://arxiv.org/pdf/2604.08011">Beyond Dense Connectivity: Explicit Sparsity for Scalable Recommendation</a></td>
    </tr>
  </tbody>
</table>

## Training Paradigm

UniRank replaces the commonly used **newest-impression supervision** paradigm with **chronological point-wise autoregressive supervision**. The difference is not the prediction loss itself—both can use point-wise binary objectives—but how a user sequence is converted into supervised targets.

<p align="center">
  <img width="1000" alt="Traditional newest-impression supervision and UniRank point-wise autoregressive supervision" src="./assets/figures/training_pipeline.png">
</p>

**Figure 1. Newest-impression supervision versus point-wise autoregressive supervision.** The upper pipeline produces supervision only for the newest target after a behavioral sequence. The lower pipeline turns successive chronological positions into targets and conditions each prediction on its preceding prefix.

| Aspect | Newest-impression supervision | UniRank point-wise autoregressive supervision |
|:--|:--|:--|
| Target construction | Select the latest impression as the supervised target for a user sequence or training window. | Treat every eligible impression or interaction anchor at position `t` as an individual supervised target. |
| Historical context | Earlier behaviors are used only as context for the final target and are commonly restricted to positive feedback. | The chronological prefix before each target is represented by item, action, and timestamp histories, preserving exposure and multi-feedback information defined by the dataset. |
| Supervision density | A sequence normally contributes one target loss. | A sequence can contribute multiple target losses at different chronological positions. |
| Sequence coverage | Intermediate impressions affect training only when they are retained as history. | Intermediate targets are learned directly and later become historical context for subsequent targets. |
| History length | Training is concentrated around the history available for the newest target. | The model is trained across short, medium, and long prefixes generated by different target positions. |

Traditional single-task **New Impression Only (newest-impression)** supervision predicts each new impression from a fixed positive-feedback history. Multiple new-impression samples may therefore share the same historical context, and each sample contributes only its single-task target.

UniRank instead uses **chronological point-wise autoregressive** supervision. Every eligible chronological target can contribute supervision for each feedback task, conditioned on the complete causal history before that target, including positive and implicit-negative feedback. The target itself and all future events are excluded; the dataloader truncates or pads the causal prefix to `max_len`.

### Why point-wise autoregressive supervision?

- **Denser supervision:** one chronological sequence provides many training targets instead of only its final impression, improving utilization of expensive interaction logs.
- **Direct learning from intermediate decisions:** impressions and feedback at earlier positions contribute their own losses rather than serving only as auxiliary context.
- **Action-aware preference evolution:** an event's multi-feedback action becomes context for later targets, allowing the model to learn how different actions change subsequent ranking outcomes.
- **Coverage across history lengths:** the same framework trains on prefixes of different lengths, reducing dependence on a single fixed newest-impression context.
- **Causal alignment:** every prediction uses only information available before its target time, matching autoregressive sequence modeling without introducing future history tokens.
- **Unified ranking and sequence learning:** target prediction and historical representation are trained from the same chronological sample organization, which is especially useful for models that jointly perform sequence modeling and feature interaction.

This paradigm produces more correlated samples from the same user and can increase training volume, so chronological splitting, user-aware metrics, class imbalance handling, and blocked data loading remain important parts of the benchmark.

## Evaluation Protocol

The sample organization must be paired with an evaluation protocol that preserves time. UniRank therefore uses a **chronological evaluation protocol** rather than treating a user-disjoint split as the only benchmark setting.

<p align="center">
  <img width="1050" alt="User-disjoint split versus chronological per-user split" src="./assets/figures/test_pipeline.png">
</p>

**Figure 2. User-disjoint versus chronological evaluation.** In a user-disjoint split, each user belongs to only one split. In a chronological split, targets are ordered by time: earlier interactions are used for training, a later interval for validation, and the final interval for testing.

| Aspect | User-disjoint protocol | UniRank chronological protocol |
|:--|:--|:--|
| Split unit | Users are assigned exclusively to train, validation, or test. | Each user's ordered targets are split into the earliest 80% for training, the next 10% for validation, and the latest 10% for testing. |
| User overlap | Validation and test users are unseen during training. | A recurring user may appear in multiple splits, but test targets occur after training targets. |
| Primary question | Can the model generalize to entirely unseen users? | Can the model rank future interactions for users under a later data distribution? |
| Historical information | Test users have no training-period model history unless a separate cold-start history policy is defined. | Previously observed interactions can form causal context for later targets according to the preprocessing rules. |
| Online interpretation | Cold-start or new-user recommendation. | Warm-start ranking for recurring users and future traffic. |

### Why chronological evaluation?

- **Matches the deployment direction of time:** the model is trained on the past and evaluated on later interactions instead of mixing future events into the training period.
- **Aligns with point-wise autoregressive supervision:** every validation or test target is evaluated from its preceding prefix under the same causal sample definition used during training.
- **Measures recurring-user ranking:** industrial rankers frequently serve users with existing histories, so retaining users across time boundaries evaluates how models exploit those histories.
- **Exposes temporal distribution shift:** changes in item supply, user intent, context, and feedback rates remain visible between training and test periods.
- **Preserves realistic target histories:** later targets can have histories accumulated before the split boundary, rather than forcing all test users into an artificial no-history condition.

Chronological evaluation is not universally superior to a user-disjoint protocol. It measures warm-start future ranking, whereas user-disjoint evaluation measures cold-start generalization; UniRank's reported results should be interpreted according to the former objective.

For a task with no positive sample in a candidate validation or test interval, preprocessing moves the boundary to the nearest valid split position without reversing chronological order. When public timestamps are unavailable, as in QK-Video, the released event order is preserved as the chronological fallback.

During DDP validation and testing, every rank performs inference on its assigned blocks. Predictions and labels are gathered across ranks before metrics are computed, so the reported **binary Logloss** (lower is better) and **global AUC** (higher is better) cover the complete evaluated split. Validation selects the monitored checkpoint; testing runs after training with the best checkpoint.

## Framework Workflow

An experiment passes through the following components:

1. **Dataset preprocessing** converts raw events into chronological samples, multi-task labels, full user histories, item side information, metadata, and block manifests.
2. **Dataset configuration** declares paths, feature types, vocabulary sizes, label columns, and blocked-loading options in `config/dataset_config.yaml`.
3. **Feature processing** builds or loads the feature map and assigns sparse embeddings to user, item, context, and action features.
4. **Action-aware loading** reads matching `data`, `user_info`, and `item_info` blocks and constructs target-specific history tensors.
5. **Model interaction** maps fields and histories into model-specific tokens, then applies either stacked unified interaction or layer-wise unified interaction.
6. **Multi-task prediction** produces one probability per label and optimizes the configured binary losses.
7. **Distributed evaluation** aggregates all rank outputs and reports logloss and AUC per task.

The main entry point is `run_expid.py`. Model and dataset selection is configuration-driven, so the same training/evaluation loop can be reused across architectures without model-specific runner scripts.

## Architecture Design

UniRank groups the registered architectures by how unified interaction is organized across the network:

| Paradigm | Description | Models |
|:--|:--|:--|
| Stacked Unified Interaction | Sequence modeling and feature interaction are arranged as consecutive modules. The sequence modeling module first extracts representations from the behavioral history; its output is then combined with user, target-item, and context features and processed by the following feature interaction module. | HiFormer, RankMixer, Zenith, TokenMixer, UniMixer, HeMix, SSR |
| Layer-wise Unified Interaction | Sequence modeling and feature interaction are integrated within each layer. Behavioral sequences and non-sequential features are processed together and updated layer by layer throughout the interaction network. | OneTrans, HyFormer, MixFormer, INFNet, EST, TokenFormer, LONGER, UltraHSTU |

The distinction concerns how sequence modeling and feature interaction modules are organized rather than which operator they use. Stacked models place the two modules in sequence, whereas layer-wise models integrate both operations into each network layer. Transformer attention, target attention, MLP mixers, sparse interaction, and hybrid dense-sequential blocks remain model-specific; input semantics, tasks, splits, and evaluation stay aligned.

## Engineering Optimizations

UniRank includes engineering support for model memory, computation, distributed execution, data access, and multi-task evaluation. These components are shared by the registered models so that architecture comparisons do not require separate training stacks.

### Memory efficiency

- **bf16 mixed precision** runs compatible forward operators under `torch.autocast`, reducing activation storage and Tensor Core compute cost while keeping binary-cross-entropy loss evaluation in FP32 for numerical compatibility.
- **Activation checkpointing** wraps each model's main interaction block with PyTorch non-reentrant checkpointing. Intermediate activations are recomputed during backward instead of being retained for the entire step. The feature is compatible with the current DDP path and is enabled only by Ultra configs by default.
- **Gradient accumulation** decouples the effective batch size from the per-step batch size, allowing large experiments to fit within device memory without changing the optimization batch semantics.
- **CPU evaluation gathering** uses a dedicated Gloo process group for serialized predictions, labels, and group IDs. This avoids transferring large `gather_object` byte tensors through NCCL and prevents evaluation aggregation from creating an unnecessary CUDA-memory peak.

### Training throughput

- **`torch.compile` acceleration** uses the Inductor backend to compile trainable dense child modules while leaving sparse embedding modules outside the compiled region. This keeps the embedding path compatible with sparse optimization and lets supported interaction blocks benefit from graph and kernel optimization. It is controlled by `enable_torch_compile` and is enabled by default in the current framework.
- **Flash Attention through SDPA** is available to models implemented with `torch.nn.functional.scaled_dot_product_attention`. When tensor dtype, shape, mask, and GPU capability satisfy PyTorch's backend constraints, SDPA can dispatch to a fused Flash Attention kernel instead of materializing the full attention matrix. OneTrans, HiFormer, LONGER, Zenith, MixFormer, HeMix, INFNet, EST, and HyFormer contain SDPA-based attention paths.
- **Flex Attention** is used by TokenFormer and UltraHSTU for structured attention patterns that require model-specific masking. `create_block_mask` constructs the block mask and `flex_attention` applies it without replacing the model's masking semantics with a dense generic attention path.
- **Separate dense and sparse optimization** applies AdamW at `1e-4` to dense network parameters and Adagrad at `0.05` to sparse embedding parameters.
- **Pinned-memory loading** and batched Parquet iteration overlap host-to-device transfer with model execution and avoid materializing the full training split in memory.
- **Distributed data parallelism** uses one CUDA process per GPU and NCCL gradient synchronization. Validation and testing are also partitioned across ranks rather than being repeated entirely on rank 0.

### Blocked data pipeline

Each split can be stored as matching block triplets:

```text
train/
+-- data/part-00000.parquet
+-- user_info/part-00000.parquet
+-- item_info/part-00000.parquet
```

The loader pairs blocks by part ID, streams Parquet batches, keeps a bounded side-information cache, and assigns whole blocks across DDP ranks using estimated sample cost. This avoids loading the complete dataset into host memory and allows preprocessing output to be consumed directly by training. Block-local user/item indices also keep side-information lookup tables bounded by the active block rather than the full dataset cardinality.

## Repository Structure

```text
UniRank/
+-- config/
|   +-- dataset_config.yaml       # Paths, schemas, labels, vocabularies, blocked loading
|   +-- model_config.yaml         # Experiment IDs, model sizes, optimization and metrics
+-- data/
|   +-- QK_Video/                 # QK-Video preprocessing and statistics
|   +-- KuaiRand/                 # KuaiRand preprocessing and statistics
|   +-- TAAC2025/                 # TencentGR preprocessing, conversion and statistics
|   +-- Taobao/                   # Taobao preprocessing and statistics
|   +-- MerRec/                   # MerRec download, preprocessing and statistics
|   +-- dataset_stats_utils.py
+-- model_zoo/                    # Fifteen registered ranking architectures
+-- unirank/                      # Training, feature, metric and shared utilities
|   +-- utils.py                  # Configuration, Parquet and DataFrame utilities
|   +-- pytorch/
|       +-- torch_utils.py        # Device, distributed and tensor utilities
|       +-- dataloaders/
|           +-- unirank_dataloader.py  # Blocked action-aware sequence loader
|           +-- rank_dataloader.py     # Train/validation/test iterator builder
+-- assets/figures/               # README and benchmark figures
+-- benchmark/                    # Accuracy logs and engineering benchmark utilities
+-- checkpoints/                  # Saved model checkpoints
+-- run_expid.py                  # Single-experiment entry point
+-- run_all.sh                    # Batch experiment launcher
+-- run_param_tuner.py            # Hyperparameter tuning entry point
+-- autotuner.py
+-- requirements.txt
+-- README.md
```

## Datasets

The paper reports the following statistics after preprocessing:

| Dataset | Instances | Users | Items | Fields | Tasks | Avg. length | Max. length |
|:--|--:|--:|--:|--:|--:|--:|--:|
| QK-Video | 493,306,303 | 4,996,176 | 3,752,235 | 10 | 4 | 99 | 6,013 |
| KuaiRand | 323,464,444 | 27,285 | 32,038,725 | 40 | 6 | 11,855 | 228,030 |
| TAAC-25 | 757,207,146 | 7,706,778 | 15,707,425 | 30 | 2 | 98 | 100 |
| Taobao | 23,601,301 | 470,570 | 831,643 | 23 | 4 | 50 | 3,756 |
| MerRec | 172,304,959 | 1,697,072 | 42,577,610 | 20 | 5 | 102 | 26,576 |

Dataset semantics and repository entry points are aligned as follows:

<table>
  <thead>
    <tr>
      <th width="15%">Dataset ID</th>
      <th width="18%">Feedback tasks</th>
      <th width="35%">Time range and construction</th>
      <th width="10%">Raw data</th>
      <th width="12%">Preprocessed data</th>
      <th width="10%">Preprocessing script</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>QK_Video_Action</code></td>
      <td>click, follow, like, share</td>
      <td>September 17--December 7, 2021; public timestamps are removed, so released order is preserved.</td>
      <td><a href="https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html">QK-Video</a></td>
      <td><a href="https://huggingface.co/datasets/salmon1802/QK-Video">QK_Video</a></td>
      <td><a href="./data/QK_Video/preprocess_QK_seq_action.py">Script</a></td>
    </tr>
    <tr>
      <td><code>KuaiRand_Video_Action</code></td>
      <td>click, follow, like, comment, forward, long-view</td>
      <td>April 8--May 8, 2022; events are ordered chronologically.</td>
      <td><a href="https://kuairand.com/">KuaiRand</a></td>
      <td><a href="https://huggingface.co/datasets/salmon1802/KuaiRand">KuaiRand</a></td>
      <td><a href="./data/KuaiRand/preprocess_Kuairand_seq_action.py">Script</a></td>
    </tr>
    <tr>
      <td><code>TencentGR_10M_Action</code></td>
      <td>click, conversion</td>
      <td>De-identified TAAC 2025 data; dates are undisclosed, users without positive feedback are removed, and conversion implies click. The paper names the processed benchmark TAAC-25.</td>
      <td><a href="https://huggingface.co/datasets/TAAC2025/TencentGR-10M">TAAC2025</a></td>
      <td><a href="https://huggingface.co/datasets/salmon1802/TAAC-25">TAAC-25</a></td>
      <td><a href="./data/TAAC2025/preprocess_TAAC2025_seq_action.py">Script</a></td>
    </tr>
    <tr>
      <td><code>Taobao_Action</code></td>
      <td>click, cart, favorite, buy</td>
      <td>Ad impressions from May 6--13, 2017 are joined with behaviors from April 22--May 13 using the preceding 24-hour window.</td>
      <td><a href="https://tianchi.aliyun.com/dataset/56">Taobao Ad Display/Click Data</a></td>
      <td><a href="https://huggingface.co/datasets/salmon1802/Taobao">Taobao</a></td>
      <td><a href="./data/Taobao/preprocess_Taobao_seq_action.py">Script</a></td>
    </tr>
    <tr>
      <td><code>MerRec_Action</code></td>
      <td>like, cart, offer, checkout, purchase</td>
      <td>May--October 2023; the current benchmark preprocessing uses the October partition.</td>
      <td><a href="https://huggingface.co/datasets/mercari-us/merrec">Mercari MerRec</a></td>
      <td><a href="https://huggingface.co/datasets/salmon1802/MerRec">MerRec</a></td>
      <td><a href="./data/MerRec/preprocess_MerRec_seq_action.py">Script</a></td>
    </tr>
  </tbody>
</table>

Features are organized into user, context, sequence, and action groups. Low-frequency categorical values map to a shared OOV ID, multi-value categorical fields use masked average pooling, and dense numeric fields are bucketized as categorical inputs.

## Installation

```bash
conda create -n UniRank python=3.10
conda activate UniRank

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The CUDA build must match the driver and GPUs on the training machine. Long-sequence acceleration paths may require a recent PyTorch/CUDA combination beyond the minimum environment above.

## Quick Start

The recommended workflow is to download a ready-to-use dataset from the [preprocessed dataset repositories](#datasets) instead of rebuilding it from raw events.

### 1. Download data

Download the required dataset to local storage, for example:

```bash
hf download salmon1802/KuaiRand \
  --repo-type dataset \
  --local-dir /path/to/data/KuaiRand_Video_Action
```

Update the corresponding local paths in `config/dataset_config.yaml`. The released data already matches the configured features, vocabularies, and task order. Raw-data preprocessing under `data/` is only needed when reproducing the dataset construction.

### 2. Train with `run_all.sh`

In `run_all.sh`, set the visible GPUs and process count, then uncomment the experiments to run:

```bash
chmod +x run_all.sh
./run_all.sh
```

Experiment definitions are in `config/model_config.yaml`. `run_all.sh` launches the selected experiments sequentially with DDP and writes their logs under `logs/`.

## Configuration Guide

### `dataset_config.yaml`

- `feature_cols`: feature name, type, source, dtype, sequence length, and vocabulary size.
- `label_col`: ordered binary task labels returned to the model.
- `train_data`, `valid_data`, `test_data`: split-specific Parquet paths.
- `*_user_info`, `*_item_info`: side-information paths paired with split data blocks.
- `blocked`, `block_cache_size`: large-data loading and caching behavior.

### `model_config.yaml`

- `model`, `dataset_id`: implementation and dataset binding.
- `num_tasks`, `task`, `loss`: task-head and objective definitions.
- `metrics`, `group_id`, `monitor`: evaluation and checkpoint-selection rules.
- `max_len`, `token_dim`, `num_layers`: history and interaction capacity.
- `dense_optimizer`, `sparse_optimizer`: separate optimization for network and embeddings.
- `batch_size`, `accumulation_steps`: effective batch-size controls.
- `gradient_checkpointing`: activation-memory trade-off during training.
- `enable_torch_compile`: enable or disable Inductor compilation of eligible dense modules.

The `Base` section provides shared defaults; each experiment overrides only the parameters needed by a model/dataset combination.

## Extending UniRank

### Add a model

1. Implement the model in `model_zoo/YourModel.py` using the shared feature map and multi-task interface.
2. Export it from `model_zoo/__init__.py`.
3. Add model/dataset experiments to `config/model_config.yaml`.
4. Reuse `unirank/pytorch/dataloaders/unirank_dataloader.py` unless the architecture requires a genuinely different input contract.
5. Add the experiment ID to `run_all.sh` and verify single-GPU and DDP execution.

All current models expose their main interaction block through the shared activation-checkpoint helper, so new large models should do the same when practical.

### Add a dataset

1. Produce chronological `train`, `valid`, and `test` samples with explicit label semantics.
2. Generate matching user histories and item side information for every block.
3. Reserve categorical ID `0` for padding/unknown values and record vocabulary sizes.
4. Write `meta_data.json` and, for blocked output, `block_manifest.json`.
5. Add a dataset entry to `config/dataset_config.yaml` and a statistics script alongside the preprocessor.
6. Add one nearby experiment configuration per model to keep cross-dataset comparisons organized.

## Reproducibility Notes

- Compare models on identical generated dataset files, not only identical raw sources.
- Keep label construction windows and action-token rules fixed across experiments.
- Report binary Logloss and global AUC for every configured task under the default protocol.
- Preserve the per-user chronological 80%/10%/10% split and any positive-aware boundary adjustments when evaluating warm-start future ranking.
- Use several seeds for small reported gains, especially when model differences are below normal run-to-run variation.
- Record model size, sequence length, token dimension, batch size, precision, and GPU count together with accuracy metrics.

## Acknowledgement

UniRank is built on top of, and deeply inspired by, the excellent [FuxiCTR](https://github.com/reczoo/FuxiCTR) project. We sincerely thank the FuxiCTR authors and contributors for their open-source work on reproducible CTR and ranking model research.

## License

This project is released under the [Apache License 2.0](./LICENSE).
