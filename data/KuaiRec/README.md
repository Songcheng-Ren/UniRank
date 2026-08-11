# KuaiRec support

This directory converts the official KuaiRec release into UniRank's blocked
sequential-action layout. The binary target is `high_watch`, defined as
`watch_ratio > 2.0`, following the binary-label example in the dataset README.

## Download

```bash
bash data/KuaiRec/download.sh
```

## Protocol A: chronological big matrix

This protocol uses only `big_matrix.csv` and splits its dates chronologically.

```bash
python data/KuaiRec/preprocess_KuaiRec_seq_action.py \
  --data_dir ./data/KuaiRec/raw \
  --output_dir /mnt/ceph-nj1-csp/bingoozhang/salmonli/data/KuaiRec_Big_Watch_Action \
  --protocol big_chrono \
  --dataset_id KuaiRec_Big_Watch_Action
```

## Protocol B: fully-observed dense test

This protocol uses `big_matrix.csv` for train/validation and reserves
`small_matrix.csv` for test. Train and validation histories never contain
small-matrix events. Test histories are built by true timestamp order.

```bash
python data/KuaiRec/preprocess_KuaiRec_seq_action.py \
  --data_dir ./data/KuaiRec/raw \
  --output_dir /mnt/ceph-nj1-csp/bingoozhang/salmonli/data/KuaiRec_Dense_Watch_Action \
  --protocol official_dense \
  --dataset_id KuaiRec_Dense_Watch_Action
```

Each output directory contains `dataset_config_snippet.yaml` with exact
vocabulary sizes. Prefer those exact sizes over the conservative upper bounds
in the repository configuration.

## Inspect and run

```bash
python data/KuaiRec/stat_KuaiRec_dataset.py \
  --output_dir /mnt/ceph-nj1-csp/bingoozhang/salmonli/data/KuaiRec_Big_Watch_Action

torchrun --standalone --nproc_per_node=4 run_expid.py \
  --config ./config \
  --expid QFormerCross3_KuaiRec_Big_Watch_Action \
  --gpu 0,1,2,3
```
