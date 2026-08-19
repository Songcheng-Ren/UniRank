# Taobao preprocessing

## Pure CTR pipeline

The CTR variant keeps every ad impression, uses only `raw_sample.clk` as
`is_click`, and encodes history actions as `exposure` or `click`. It does not
read `behavior_log`.

```bash
python data/Taobao/preprocess_Taobao_ctr.py \
  --data_dir ./data/Taobao/raw \
  --output_dir /mnt/ceph-nj1-csp/bingoozhang/salmonli/data/Taobao_CTR
```

Run QFormerCross15 or QFormerCross16 with:

```bash
torchrun --standalone --nproc_per_node=4 run_expid.py \
  --config ./config \
  --expid QFormerCross15_Taobao_CTR \
  --gpu 0,1,2,3
```

The existing `Taobao_Action` multi-task pipeline is unchanged and remains the
default mode of `preprocess_Taobao_seq_action.py`.
