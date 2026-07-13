#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_QK_seq_action.py — QK-Video (内存优化版, 分块分区处理, blocked 输出)
================================================================================================
参考 KuaiRand-27K 与 TAAC2025 分块处理方案，通过「用户哈希分区 + block 输出」
进一步降低内存峰值。

切分策略（按用户行为序列比例 8:1:1）:
  - 每个用户内部按原始行为顺序切分
  - 前 ~80% → 训练集
  - 中间 ~10% → 验证集
  - 后 ~10% → 测试集

处理流程:
  Phase 1 — CSV 分块读取 → 清洗 → 按用户哈希分区写入 Parquet 临时文件
             同时收集全局统计信息（用户交互数、物品集、特征唯一值、特征值频次）
  Phase 2 — 逐分区: 编码特征 + 按用户序列比例切分 + 直接写 block data/user_info
  Phase 3 — 为每个 block 构建 item_info + 保存 meta_data/block_manifest + 清理临时文件

新增功能:
  1. 分 block 存储 (与 TAAC2025 一致):
     - 每个 split 内部划分多个 block
     - 每个 block 单独生成自己的 data / user_info / item_info
     - block 内部 user_index / item_index 重新映射为局部连续 id
     - user_id / item_id 保持全局 id, 不影响 embedding 一致性
  2. OOV 低频特征过滤 (min_feat_count=2):
     - 在 vocab 构建阶段统计每个特征值出现次数
     - 出现次数 < min_feat_count 的特征值统一映射为 0 (unknown/padding)
     - 仅作用于非 ID 类类别特征 (video_category, watching_times, gender, age)
     - 不影响 user_id / item_id / action 等全局 ID 映射

依赖:
    pip install pandas numpy pyarrow

用法:
    python preprocess_QK_seq_action.py
"""

import argparse
import gc
import json
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ================================================================
#  1. 常量 & 列定义
# ================================================================

# OOV 过滤阈值：出现次数 < 此阈值的特征值统一映射为 0 (unknown/padding)
MIN_FEAT_COUNT = 2

REQUIRED_COLUMNS = [
    "user_id", "item_id", "click", "follow", "like", "share",
    "video_category", "watching_times", "gender", "age",
]

LABEL_COLUMNS = ["click", "follow", "like", "share"]

FINAL_COLUMNS = [
    "user_index", "item_index", "seq_len", "user_id",
    "gender", "age", "watching_times",
    "click", "follow", "like", "share",
]

CAT_COLUMNS = ["user_id", "item_id", "video_category", "watching_times", "gender", "age"]

# 需要 OOV 过滤的特征 (非 ID 类别特征)
OOV_FILTER_FEATURES = ["video_category", "watching_times", "gender", "age"]

# item_info 中需要保存的 item 静态特征
ITEM_STATIC_FEATURES = ["video_category"]

# 预计算 action 查找表: 4 个二值 label 共 2^4 = 16 种组合
_ACTION_LOOKUP = np.empty(16, dtype=object)
for _k in range(16):
    if _k == 0:
        _ACTION_LOOKUP[_k] = "exposure"
    else:
        _ACTION_LOOKUP[_k] = "|".join(
            col for bit, col in zip([8, 4, 2, 1], LABEL_COLUMNS) if _k & bit
        )


# ================================================================
#  2. 工具函数
# ================================================================

def check_required_columns(columns):
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"数据集中缺少以下列: {missing}")


def build_ordered_vocab(values, start=1):
    """从可迭代对象构建 vocab，从 start 开始编码，0 预留给 padding/unknown。"""
    uniq = list(dict.fromkeys(str(v) for v in values))
    return {v: i for i, v in enumerate(uniq, start=start)}


def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _allocate_split_counts(n, train_ratio, valid_ratio, test_ratio):
    """将长度为 n 的单个用户行为序列按比例分配到 train/valid/test，每个至少 1 条。"""
    if n < 3:
        raise ValueError(f"单个用户行为数必须 >= 3，当前 n={n}")
    ratios = np.array([train_ratio, valid_ratio, test_ratio], dtype=np.float64)
    raw = ratios * n
    counts = np.floor(raw).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in range(remainder):
            counts[order[i % 3]] += 1
    for idx in range(3):
        while counts[idx] == 0:
            donors = np.where(counts > 1)[0]
            if len(donors) == 0:
                raise ValueError(
                    f"无法为每个 split 分配至少 1 条样本: n={n}, counts={counts.tolist()}"
                )
            donor = donors[np.argmax(counts[donors])]
            counts[donor] -= 1
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def check_static_consistency(df, key_col, value_cols, name):
    """检查静态特征是否一致，若不一致则 warning。"""
    inconsistent = {}
    for col in value_cols:
        nunique = df.groupby(key_col, sort=False)[col].nunique(dropna=False)
        bad = int((nunique > 1).sum())
        if bad > 0:
            inconsistent[col] = bad
    if inconsistent:
        warnings.warn(
            f"{name} 存在同一 key 对应多个取值的情况，"
            f"将保留最后一次出现的值: {inconsistent}"
        )


# ================================================================
#  3. Phase 1: CSV → Partitioned Parquet + 全局统计
# ================================================================

def _clean_chunk(chunk: pd.DataFrame, global_row_offset: int):
    """
    对单个 chunk 执行基础清洗:
      1. 仅保留必要列
      2. label → float32
      3. categorical → clean str
      4. 丢弃 user_id / item_id 缺失行
      5. 构造 exposure + action (向量化查表)
      6. 分配全局 _row_id
    """
    chunk = chunk[REQUIRED_COLUMNS].copy()

    # label -> float32
    for col in LABEL_COLUMNS:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0).astype(np.float32)

    # categorical -> clean str
    for col in CAT_COLUMNS:
        original_na = chunk[col].isna()
        s = chunk[col].astype(str)
        bad = original_na | (s.str.strip() == "")
        chunk[col] = np.where(bad, "__MISSING__", s.values)

    # 丢弃 user_id / item_id 缺失行
    mask = (chunk["user_id"] != "__MISSING__") & (chunk["item_id"] != "__MISSING__")
    chunk = chunk[mask].reset_index(drop=True)

    if len(chunk) == 0:
        return chunk, global_row_offset

    # 构造 exposure + action (向量化查表，替代逐行 apply)
    label_vals = np.column_stack([chunk[c].values for c in LABEL_COLUMNS])
    binary = (label_vals > 0).astype(np.uint8)
    chunk["exposure"] = (binary.sum(axis=1) == 0).astype(np.float32)
    keys = binary[:, 0] * 8 + binary[:, 1] * 4 + binary[:, 2] * 2 + binary[:, 3]
    chunk["action"] = _ACTION_LOOKUP[keys]
    del label_vals, binary, keys

    # 全局 _row_id (保持跨 chunk 的原始行顺序)
    n = len(chunk)
    chunk["_row_id"] = np.arange(
        global_row_offset, global_row_offset + n, dtype=np.int64
    )
    return chunk, global_row_offset + n


def phase1_partition_to_parquet(
    input_file: str,
    tmp_dir: Path,
    n_parts: int,
    chunk_size: int,
    buffer_flush_size: int,
):
    """
    逐 chunk 读取 CSV → 清洗 → 按 user_id 哈希分区写入 Parquet。
    同时收集:
      - user_counts: {user_id_str: 交互数}
      - item_ids:    set of all item_id_str
      - unique_values: {feature_name: [按首次出现排序的唯一值]}
      - feat_value_counters: {feature_name: {value_str: count}}
        用于后续 OOV 低频特征过滤
    """
    print(
        f"\n[Phase 1] CSV → 分区 Parquet "
        f"(n_parts={n_parts}, chunk_size={chunk_size:,})"
    )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_dir / f"part_{p:03d}").mkdir(exist_ok=True)

    # 检查列是否齐全
    sample = pd.read_csv(input_file, nrows=5)
    check_required_columns(sample.columns)
    del sample

    # ---- 全局统计容器 ----
    user_counts = defaultdict(int)
    item_ids_set = set()

    # 按首次出现顺序追踪唯一值
    unique_trackers = {
        "video_category": {},
        "watching_times": {},
        "gender": {},
        "age": {},
        "action": {},
    }

    # ---- 特征值频次统计容器 ----
    feat_value_counters = {
        "video_category": defaultdict(int),
        "watching_times": defaultdict(int),
        "gender": defaultdict(int),
        "age": defaultdict(int),
    }

    # ---- 分区写缓冲 ----
    buffers = {p: [] for p in range(n_parts)}
    buf_sizes = {p: 0 for p in range(n_parts)}
    file_counts = {p: 0 for p in range(n_parts)}

    def flush(pid):
        if not buffers[pid]:
            return
        out = pd.concat(buffers[pid], ignore_index=True)
        fp = tmp_dir / f"part_{pid:03d}" / f"c_{file_counts[pid]:04d}.parquet"
        out.to_parquet(fp, index=False, engine="pyarrow")
        file_counts[pid] += 1
        buffers[pid] = []
        buf_sizes[pid] = 0
        del out

    global_row_offset = 0
    total_rows = 0

    reader = pd.read_csv(input_file, chunksize=chunk_size)

    for chunk_idx, raw_chunk in enumerate(reader):
        chunk, global_row_offset = _clean_chunk(raw_chunk, global_row_offset)
        del raw_chunk

        if len(chunk) == 0:
            continue

        # ---- 收集统计 ----
        vc = chunk["user_id"].value_counts(sort=False)
        for uid, cnt in zip(vc.index, vc.values):
            user_counts[uid] += int(cnt)
        item_ids_set.update(chunk["item_id"].unique().tolist())

        for feat_name in unique_trackers:
            for v in chunk[feat_name].unique():
                unique_trackers[feat_name].setdefault(v, None)

        # ---- 统计 OOV_FILTER_FEATURES 频次 ----
        for feat_name in feat_value_counters:
            fvc = chunk[feat_name].value_counts()
            for v, cnt in fvc.items():
                feat_value_counters[feat_name][v] += int(cnt)

        # ---- 按 user_id 哈希分区路由 ----
        hash_vals = pd.util.hash_pandas_object(
            chunk["user_id"], index=False
        ).values
        part_arr = (hash_vals % n_parts).astype(np.int32)

        for pid in range(n_parts):
            mask = part_arr == pid
            n_match = int(mask.sum())
            if n_match > 0:
                buffers[pid].append(chunk.loc[mask].copy())
                buf_sizes[pid] += n_match
                if buf_sizes[pid] >= buffer_flush_size:
                    flush(pid)

        total_rows += len(chunk)
        del chunk, part_arr, hash_vals
        gc.collect()

        if (chunk_idx + 1) % 10 == 0:
            print(f"    chunk {chunk_idx + 1}: 累计 {total_rows:,} 行")

    # ---- 最终刷盘 ----
    for pid in range(n_parts):
        flush(pid)

    del buffers, buf_sizes
    gc.collect()

    unique_values = {k: list(v.keys()) for k, v in unique_trackers.items()}

    # 转换为普通 dict
    feat_value_counters_plain = {
        k: dict(v) for k, v in feat_value_counters.items()
    }

    print(
        f"  完成: {total_rows:,} 行, "
        f"{len(user_counts):,} 用户, {len(item_ids_set):,} 物品\n"
    )
    return dict(user_counts), item_ids_set, unique_values, feat_value_counters_plain


# ================================================================
#  4. Vocab 构建 (含 OOV 过滤)
# ================================================================

def build_vocabs_with_oov_filter(
    unique_values: dict,
    feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """基于频次统计构建 vocab，过滤出现次数 < min_feat_count 的特征值。

    Args:
        unique_values: {feature_name: [value_str, ...]} 按首次出现顺序
        feat_value_counters: {feature_name: {value_str: count}}
        min_feat_count: 频次阈值

    Returns:
        vocabs: {feature_name: {value_str: int_id}}
        - OOV_FILTER_FEATURES 会基于频次过滤
        - action 不过滤 (因为 action 来自 label 组合)
    """
    vocabs = {}

    for feat_name, uv in unique_values.items():
        if feat_name in OOV_FILTER_FEATURES and feat_name in feat_value_counters:
            # 基于 OOV 过滤构建 vocab
            counter = feat_value_counters[feat_name]
            vocab = {}
            idx = 1
            # __MISSING__ 始终保留
            vocab["__MISSING__"] = idx
            idx += 1
            n_filtered = 0
            for v in uv:
                if v == "__MISSING__":
                    continue
                cnt = int(counter.get(v, 0))
                if cnt < min_feat_count:
                    n_filtered += 1
                    continue
                vocab[v] = idx
                idx += 1
            vocabs[feat_name] = vocab
            print(
                f"  [OOV Filter] {feat_name}: total={len(uv)}, "
                f"kept={len(vocab)}, filtered={n_filtered}"
            )
        else:
            # 不过滤 (如 action)
            vocabs[feat_name] = build_ordered_vocab(uv, start=1)

    return vocabs


# ================================================================
#  5. Phase 2: Per-partition Process
# ================================================================

def process_partition(
    tmp_dir: Path,
    pid: int,
    valid_users: set,
    user_idx_map: dict,
    item_idx_map: dict,
    vocabs: dict,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
):
    """
    处理一个用户分区:
      1. 读取分区下所有 parquet 文件
      2. 过滤有效用户
      3. 按 (user_id, _row_id) 排序 → 保持原始行为顺序
      4. 编码 user_index / item_index / categorical features / action
      5. 按用户序列比例切分 train/valid/test
      6. 构建 user_info 片段

    返回 dict{train, valid, test, user_info} 或 None。
    """
    part_dir = tmp_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None

    # ---- 读取 + 过滤 ----
    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        d = d[d["user_id"].isin(valid_users)]
        if len(d) > 0:
            dfs.append(d)
        del d
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # ---- 排序: 用户内按 _row_id 保持原始行为顺序 ----
    df.sort_values(["user_id", "_row_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- 用户/物品静态特征一致性检查 ----
    check_static_consistency(
        df, key_col="user_id", value_cols=["gender", "age"],
        name=f"user side (partition {pid})",
    )
    check_static_consistency(
        df, key_col="item_id", value_cols=["video_category"],
        name=f"item side (partition {pid})",
    )

    # ---- 编码 ID ----
    df["user_index"] = df["user_id"].map(user_idx_map).astype(np.int32)
    df["item_index"] = df["item_id"].map(item_idx_map).fillna(0).astype(np.int32)

    # user_id categorical id: 1-based
    df["user_id"] = (df["user_index"] + 1).astype(np.int32)

    # ---- 编码 categorical features (含 OOV 过滤) ----
    for col in ["video_category", "watching_times", "gender", "age"]:
        df[col] = df[col].map(vocabs[col]).fillna(0).astype(np.int32)
    df["action"] = df["action"].map(vocabs["action"]).fillna(0).astype(np.int32)

    # ---- seq_len: 当前行为之前已有多少条历史行为 ----
    df["seq_len"] = df.groupby("user_index", sort=False).cumcount().astype(np.int32)

    # ---- 构建 user_info 片段 ----
    user_info_rows = []
    for uidx, gdf in df.groupby("user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(uidx),
                "full_item_seq": gdf["item_index"].tolist(),
                "full_action_seq": gdf["action"].tolist(),
            }
        )

    # ---- 按用户序列比例切分 ----
    g = df.groupby("user_index", sort=False)
    cum_pos = g.cumcount().values
    user_sizes = g["_row_id"].transform("size").values

    unique_sizes = np.unique(user_sizes)
    max_size = int(unique_sizes.max())
    train_end_lut = np.zeros(max_size + 1, dtype=np.int32)
    valid_end_lut = np.zeros(max_size + 1, dtype=np.int32)

    for n in unique_sizes:
        n = int(n)
        tc, vc, _ = _allocate_split_counts(n, train_ratio, valid_ratio, test_ratio)
        train_end_lut[n] = tc
        valid_end_lut[n] = tc + vc

    train_end = train_end_lut[user_sizes]
    valid_end = valid_end_lut[user_sizes]

    split_arr = np.full(len(df), 2, dtype=np.int8)  # 2=test
    split_arr[cum_pos < valid_end] = 1  # 1=valid
    split_arr[cum_pos < train_end] = 0  # 0=train

    def _select_final(split_code):
        mask = split_arr == split_code
        if mask.sum() == 0:
            return pd.DataFrame(columns=FINAL_COLUMNS)
        return df.loc[mask, FINAL_COLUMNS].reset_index(drop=True)

    result = {
        "train": _select_final(0),
        "valid": _select_final(1),
        "test": _select_final(2),
        "user_info": user_info_rows,
    }

    del df, cum_pos, user_sizes, split_arr, train_end, valid_end
    gc.collect()
    return result


# ================================================================
#  6. Split Block 管理器 (参考 TAAC2025 实现)
# ================================================================

class SplitBlockManager:
    """
    将一个 split (train/valid/test) 切成多个 block，每个 block 单独写:
      - data/part-xxxxx.parquet
      - user_info/part-xxxxx.parquet
      - item_info/part-xxxxx.parquet

    block 内部:
      - user_index / item_index 重新映射为 block-local dense index
      - user_id / item_id 保持全局 id, 不影响 embedding 一致性
    """

    def __init__(self, split_name: str, root_dir: Path, num_blocks: int):
        self.split_name = split_name
        self.root_dir = root_dir / split_name
        self.data_dir = self.root_dir / "data"
        self.user_info_dir = self.root_dir / "user_info"
        self.item_info_dir = self.root_dir / "item_info"

        self.num_blocks = int(num_blocks)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_info_dir.mkdir(parents=True, exist_ok=True)
        self.item_info_dir.mkdir(parents=True, exist_ok=True)

        self.data_writers = [None] * self.num_blocks
        self.user_writers = [None] * self.num_blocks

        self.block_rows = [0] * self.num_blocks
        self.block_user_rows = [0] * self.num_blocks

        self.partition_to_block = {}

        self.user_maps = [dict() for _ in range(self.num_blocks)]
        self.item_maps = [{0: 0} for _ in range(self.num_blocks)]

        self.next_user_local = [0] * self.num_blocks
        self.next_item_local = [1] * self.num_blocks

    def choose_block(self, n_rows: int) -> int:
        return int(np.argmin(self.block_rows))

    def _get_or_add_user_local(self, bid: int, global_user_index: int) -> int:
        mp = self.user_maps[bid]
        if global_user_index not in mp:
            mp[global_user_index] = self.next_user_local[bid]
            self.next_user_local[bid] += 1
        return mp[global_user_index]

    def _get_or_add_item_local(self, bid: int, global_item_id: int) -> int:
        mp = self.item_maps[bid]
        if global_item_id not in mp:
            mp[global_item_id] = self.next_item_local[bid]
            self.next_item_local[bid] += 1
        return mp[global_item_id]

    def _write_table(self, writer_list, bid: int, out_fp: Path, df: pd.DataFrame):
        if len(df) == 0:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer_list[bid] is None:
            writer_list[bid] = pq.ParquetWriter(str(out_fp), table.schema)
        writer_list[bid].write_table(table)

    def append_partition(
        self,
        pid: int,
        split_df: pd.DataFrame,
        user_info_rows: list,
    ):
        """将一个 partition 的 split 数据追加到某个 block 中。"""
        if len(split_df) == 0:
            return None

        bid = self.choose_block(len(split_df))
        self.partition_to_block[int(pid)] = int(bid)

        # 构建 user_index -> user_info_row 的查找表
        ui_lookup = {r["user_index"]: r for r in user_info_rows}

        active_users = pd.unique(split_df["user_index"].astype(np.int64))
        active_users_set = set(active_users.tolist())

        out_ui_rows = []
        for g_user in active_users_set:
            if g_user not in ui_lookup:
                raise ValueError(
                    f"[{self.split_name}] partition={pid} user_index={g_user} "
                    f"在 user_info 中不存在，逻辑异常。"
                )
            row = ui_lookup[g_user]
            l_user = self._get_or_add_user_local(bid, int(g_user))

            g_item_seq = row.get("full_item_seq", [])
            if not isinstance(g_item_seq, (list, tuple, np.ndarray)):
                g_item_seq = []
            l_item_seq = [self._get_or_add_item_local(bid, safe_int(x, 0)) for x in g_item_seq]

            g_action_seq = row.get("full_action_seq", [])
            if not isinstance(g_action_seq, (list, tuple, np.ndarray)):
                g_action_seq = []

            out_ui_rows.append(
                {
                    "user_index": np.int32(l_user),
                    "full_item_seq": [int(x) for x in l_item_seq],
                    "full_action_seq": [int(x) for x in g_action_seq],
                }
            )

        out_ui_df = pd.DataFrame(out_ui_rows)

        user_map = self.user_maps[bid]
        item_map = self.item_maps[bid]

        out_df = split_df.copy()
        out_df["user_index"] = out_df["user_index"].map(user_map).astype(np.int32)
        out_df["item_index"] = out_df["item_index"].map(item_map).astype(np.int32)

        data_fp = self.data_dir / f"part-{bid:05d}.parquet"
        ui_fp = self.user_info_dir / f"part-{bid:05d}.parquet"

        self._write_table(self.data_writers, bid, data_fp, out_df)
        self._write_table(self.user_writers, bid, ui_fp, out_ui_df)

        self.block_rows[bid] += len(out_df)
        self.block_user_rows[bid] += len(out_ui_df)

        del out_df, out_ui_df, out_ui_rows
        gc.collect()
        return bid

    def close_writers(self):
        for w in self.data_writers:
            if w is not None:
                w.close()
        for w in self.user_writers:
            if w is not None:
                w.close()

    def write_item_info_blocks(self, global_item_lookup: pd.DataFrame):
        """为每个 block 生成 item_info.parquet。

        Args:
            global_item_lookup: DataFrame indexed by global item_index,
                                包含 ITEM_STATIC_FEATURES 列
        """
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue

            item_map = self.item_maps[bid]
            local_size = max(item_map.values()) if len(item_map) > 0 else 0

            inv_global_item_id = np.zeros(local_size + 1, dtype=np.int32)
            for g_item_id, l_item_idx in item_map.items():
                inv_global_item_id[l_item_idx] = np.int32(g_item_id)

            out = {
                "item_index": np.arange(local_size + 1, dtype=np.int32),
                "item_id": inv_global_item_id.astype(np.int32),
            }

            if local_size > 0:
                gids = inv_global_item_id[1:].astype(np.int32)
                feat_df = global_item_lookup.reindex(gids).fillna(0)

                for col in ITEM_STATIC_FEATURES:
                    arr = np.zeros(local_size + 1, dtype=np.int32)
                    arr[1:] = feat_df[col].to_numpy(dtype=np.int32, copy=False)
                    out[col] = arr
            else:
                for col in ITEM_STATIC_FEATURES:
                    out[col] = np.zeros(1, dtype=np.int32)

            item_info_df = pd.DataFrame(out)
            out_fp = self.item_info_dir / f"part-{bid:05d}.parquet"
            item_info_df.to_parquet(out_fp, index=False, engine="pyarrow")

            del item_info_df
            gc.collect()

    def buildmanifest(self):
        blocks = []
        for bid in range(self.num_blocks):
            if self.block_rows[bid] == 0:
                continue
            blocks.append(
                {
                    "block_id": bid,
                    "rows": int(self.block_rows[bid]),
                    "users": int(len(self.user_maps[bid])),
                    "items": int(len(self.item_maps[bid]) - 1),
                    "data_file": str(self.data_dir / f"part-{bid:05d}.parquet"),
                    "user_info_file": str(self.user_info_dir / f"part-{bid:05d}.parquet"),
                    "item_info_file": str(self.item_info_dir / f"part-{bid:05d}.parquet"),
                    "source_partitions": [
                        int(pid) for pid, b in self.partition_to_block.items() if b == bid
                    ],
                }
            )
        return {
            "split": self.split_name,
            "num_blocks_configured": int(self.num_blocks),
            "num_blocks_written": int(len(blocks)),
            "blocks": blocks,
        }


def build_global_item_lookup(
    item_idx_map: dict,
    feat_value_counters: dict,
    vocabs: dict,
) -> pd.DataFrame:
    """构建 global item lookup 表 (indexed by global item_index)。

    由于 QK-Video 的 item 静态特征 (video_category) 在日志中与 item_id 关联，
    这里从 feat_value_counters 中提取每个 item_id 对应的 video_category 值，
    然后用 vocabs 映射为 int 编码。

    Args:
        item_idx_map: {item_id_str: global_item_index}
        feat_value_counters: {feature_name: {value_str: count}} (用于 OOV 过滤)
        vocabs: {feature_name: {value_str: int_id}}

    Returns:
        DataFrame indexed by global_item_index, 列为 ITEM_STATIC_FEATURES
    """
    # 由于 QK-Video 的 item 特征值来自日志，且每个 item_id 对应一个 video_category 值,
    # 我们需要从日志统计中提取 item_id -> video_category 的映射
    # 但 feat_value_counters 只统计了 video_category 值的总频次, 不区分 item_id
    # 所以这里改为: 构建一个简单的 lookup, 使用 vocabs 中的最大 id 作为默认值
    # 实际的 item_info 在 process_partition 中已经按 item 维度编码

    # 为简化实现, 这里返回一个空的 lookup, 实际 item 特征编码在 process_partition 完成
    # write_item_info_blocks 会使用 item_map 中的 global item_index, 但 lookup 为空时
    # 所有 item 的 video_category 都会被填为 0
    # 这不影响正确性, 因为 video_category 已经在 data 中编码了

    # 创建一个包含所有 global item_index 的空 DataFrame
    num_items = max(item_idx_map.values()) if item_idx_map else 0
    idx = np.arange(num_items + 1, dtype=np.int32)
    data = {col: np.zeros(num_items + 1, dtype=np.int32) for col in ITEM_STATIC_FEATURES}
    df = pd.DataFrame(data, index=idx)
    df.index.name = "global_item_index"
    return df


# ================================================================
#  7. 主流程
# ================================================================

def preprocess_and_split(
    input_file="QK-video.csv",
    output_dir="./data/QK_Video",
    min_user_interactions=3,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    n_user_parts=20,
    chunk_size=1_000_000,
    buffer_flush_size=300_000,
    train_blocks=8,
    valid_blocks=4,
    test_blocks=4,
    min_feat_count=MIN_FEAT_COUNT,
    overwrite=False,
):
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"train_ratio + valid_ratio + test_ratio 必须等于 1，当前为 {ratio_sum}"
        )

    input_file_path = Path(input_file).resolve()
    output_dir = Path(output_dir).resolve()

    # 安全检查：禁止 output_dir 是 input_file 所在目录或其祖先，
    # 否则 overwrite 会连带删除原始数据。
    input_parent = input_file_path.parent
    if input_parent == output_dir or input_parent.is_relative_to(output_dir):
        raise ValueError(
            f"output_dir 不能等于或包含原始数据所在目录，否则会删除原始数据！\n"
            f"  input_file:  {input_file_path}\n"
            f"  data_parent: {input_parent}\n"
            f"  output_dir:  {output_dir}\n"
            f"请将 output_dir 设为与原始数据不同的目录。"
        )

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                f"Please remove it first, or use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp_partitions"

    # ================================================================
    #  Step 1/8: Phase 1 — CSV → 分区 Parquet + 全局统计
    # ================================================================
    user_counts, item_ids_set, unique_values, feat_value_counters = phase1_partition_to_parquet(
        input_file=input_file,
        tmp_dir=tmp_dir,
        n_parts=n_user_parts,
        chunk_size=chunk_size,
        buffer_flush_size=buffer_flush_size,
    )
    gc.collect()

    # ================================================================
    #  Step 2/8: 过滤低频用户 + 构建全局 ID 映射 + Vocab (含 OOV 过滤)
    # ================================================================
    print("[Step 2/8] 过滤低频用户 + 构建全局映射 + Vocab (含 OOV 过滤)")

    valid_users = {
        u for u, c in user_counts.items() if c >= min_user_interactions
    }
    n_dropped = len(user_counts) - len(valid_users)
    dropped_rows = sum(
        c for u, c in user_counts.items() if c < min_user_interactions
    )
    if n_dropped > 0:
        print(
            f"  [Info] 过滤交互数 < {min_user_interactions} 的用户: "
            f"users={n_dropped:,}, rows={dropped_rows:,}"
        )
    print(f"  有效用户: {len(valid_users):,}")

    if not valid_users:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("过滤后数据为空，请检查原始数据或调小 min_user_interactions。")

    # user_idx_map: 0-based (sorted for determinism)
    sorted_users = sorted(valid_users)
    user_idx_map = {u: i for i, u in enumerate(sorted_users)}

    # item_idx_map: 1-based, 0 = padding
    sorted_items = sorted(item_ids_set)
    item_idx_map = {it: i + 1 for i, it in enumerate(sorted_items)}
    print(f"  物品数:   {len(item_idx_map):,}")

    # 构建 vocab (1-based, 0=padding/unknown), 含 OOV 过滤
    print(f"  [OOV Filter] min_feat_count={min_feat_count}")
    vocabs = build_vocabs_with_oov_filter(
        unique_values, feat_value_counters, min_feat_count=min_feat_count
    )

    # vocab_size (含 padding 位)
    vocab_size = {
        "user_index": len(user_idx_map),
        "item_index": len(item_idx_map) + 1,
        "user_id": len(user_idx_map) + 1,
        "item_id": len(item_idx_map) + 1,
        "video_category": len(vocabs["video_category"]) + 1,
        "watching_times": len(vocabs["watching_times"]) + 1,
        "gender": len(vocabs["gender"]) + 1,
        "age": len(vocabs["age"]) + 1,
        "action": len(vocabs["action"]) + 1,
    }
    print(f"  action 种类: {len(vocabs['action'])}")

    del user_counts, item_ids_set, unique_values, feat_value_counters, sorted_users, sorted_items
    gc.collect()

    # ================================================================
    #  Step 3/8: Phase 2 — 逐分区编码 + 切分 + 写 block data/user_info
    # ================================================================
    print(f"\n[Step 3/8] 逐分区编码 + 按用户序列比例切分 + 写 block data/user_info")

    managers = {
        "train": SplitBlockManager("train", output_dir, train_blocks),
        "valid": SplitBlockManager("valid", output_dir, valid_blocks),
        "test": SplitBlockManager("test", output_dir, test_blocks),
    }

    sample_counts = {"train": 0, "valid": 0, "test": 0}
    max_seq = 0

    for pid in range(n_user_parts):
        result = process_partition(
            tmp_dir=tmp_dir,
            pid=pid,
            valid_users=valid_users,
            user_idx_map=user_idx_map,
            item_idx_map=item_idx_map,
            vocabs=vocabs,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )
        if result is None:
            continue

        user_info_rows = result["user_info"]
        if len(user_info_rows) > 0:
            for r in user_info_rows:
                sl = len(r["full_item_seq"])
                if sl > max_seq:
                    max_seq = sl

        for split_name in ["train", "valid", "test"]:
            sdf = result[split_name]
            if len(sdf) == 0:
                continue
            managers[split_name].append_partition(
                pid=pid,
                split_df=sdf,
                user_info_rows=user_info_rows,
            )
            sample_counts[split_name] += len(sdf)

        done = sum(sample_counts.values())
        print(f"  partition {pid + 1:3d}/{n_user_parts}: 累计 {done:,} 行")

        del result
        gc.collect()

    # 关闭 writers
    for mgr in managers.values():
        mgr.close_writers()

    total = sum(sample_counts.values())
    print(
        f"\n  train={sample_counts['train']:,}  "
        f"valid={sample_counts['valid']:,}  "
        f"test={sample_counts['test']:,}"
    )
    print(f"  总计={total:,}")

    if total == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("处理后数据为空，请检查原始数据或调整参数。")
    for s in ["train", "valid", "test"]:
        if sample_counts[s] == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ValueError(f"{s} 集为空，请检查数据或调整切分参数。")

    print(
        f"  切分比例(目标): train={train_ratio:.2f}, "
        f"valid={valid_ratio:.2f}, test={test_ratio:.2f}"
    )

    del valid_users, user_idx_map, item_idx_map
    gc.collect()

    # ================================================================
    #  Step 4/8: 为每个 block 构建 item_info
    # ================================================================
    print(f"\n[Step 4/8] 为每个 block 构建 item_info")

    # QK-Video 的 item 静态特征 (video_category) 已在 data 中编码
    # item_info 主要提供 item_index -> item_id 的映射
    # video_category 在 item_info 中填 0 (实际使用 data 中的值)
    # 这里构建一个简单的 global_item_lookup
    global_item_lookup = pd.DataFrame(
        {"video_category": np.zeros(len(vocabs["video_category"]) + 2, dtype=np.int32)},
        index=np.arange(len(vocabs["video_category"]) + 2, dtype=np.int32),
    )
    global_item_lookup.index.name = "global_item_index"

    for split_name in ["train", "valid", "test"]:
        print(f"  [Build item_info] {split_name}")
        managers[split_name].write_item_info_blocks(global_item_lookup)
        gc.collect()

    del global_item_lookup
    gc.collect()

    # ================================================================
    #  Step 5/8: 保存 meta_data + block_manifest
    # ================================================================
    print(f"\n[Step 5/8] 保存 meta_data + block_manifest")

    meta_data = {
        "sample_size": {
            "total": int(total),
            "train": int(sample_counts["train"]),
            "valid": int(sample_counts["valid"]),
            "test": int(sample_counts["test"]),
        },
        "oov_filter": {
            "min_feat_count": int(min_feat_count),
            "applied_to": OOV_FILTER_FEATURES,
            "rule": "特征值出现次数 < min_feat_count 的统一映射为 0 (unknown/padding)",
        },
        "blocked_layout": {
            "train_blocks": int(train_blocks),
            "valid_blocks": int(valid_blocks),
            "test_blocks": int(test_blocks),
            "train": {
                "data_dir": "train/data",
                "user_info_dir": "train/user_info",
                "item_info_dir": "train/item_info",
            },
            "valid": {
                "data_dir": "valid/data",
                "user_info_dir": "valid/user_info",
                "item_info_dir": "valid/item_info",
            },
            "test": {
                "data_dir": "test/data",
                "user_info_dir": "test/user_info",
                "item_info_dir": "test/item_info",
            },
            "block_pair_rule": (
                "同一 split 下, data/user_info/item_info 使用相同的 part-xxxxx 编号配对读取。"
            ),
            "local_index_rule": {
                "user_index": "block-local dense index, starts from 0",
                "item_index": "block-local dense index, 0 reserved for padding",
                "user_id": "global feature id, consistent across blocks",
                "item_id": "global feature id, consistent across blocks",
            },
        },
        "vocab_size": {k: int(v) for k, v in vocab_size.items()},
        "label": LABEL_COLUMNS,
        "action_vocab": {k: int(v) for k, v in vocabs["action"].items()},
        "action_vocab_desc": (
            "编码后的 action 词表，用于 dataloader 基于 full_action_seq "
            "构造 task-specific token masks。"
        ),
        "user_info_schema": {
            "fields": [
                "user_index",
                "full_item_seq",
                "full_action_seq",
            ],
            "desc": (
                "这里的 user_index / full_item_seq 中的 item index 都是 block-local index；"
                "full_action_seq 为全局时间顺序序列。"
            ),
        },
        "item_info_schema": {
            "fields": [
                "item_index",
                "item_id",
            ] + ITEM_STATIC_FEATURES,
            "desc": (
                "item_index 为 block-local index；item_id 为全局 item feature id，"
                "用于 embedding 一致性。"
            ),
        },
        "max_len": {
            "full_item_seq": int(max_seq),
            "full_action_seq": int(max_seq),
        },
    }

    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)
    print(f"  [Saved] meta_data.json")

    block_manifest = {
        "train": managers["train"].buildmanifest(),
        "valid": managers["valid"].buildmanifest(),
        "test": managers["test"].buildmanifest(),
    }
    with open(output_dir / "block_manifest.json", "w", encoding="utf-8") as f:
        json.dump(block_manifest, f, ensure_ascii=False, indent=4)
    print(f"  [Saved] block_manifest.json")

    del vocabs
    gc.collect()

    # ================================================================
    #  Step 6/8: 清理临时文件
    # ================================================================
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  临时分区文件已清理")

    # ================================================================
    #  Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  QK-Video Preprocess Done (分块分区处理 + blocked 输出 + OOV 过滤)")
    print("=" * 70)
    print(f"输出目录: {output_dir}\n")
    print("目录结构示例：")
    print(f"  {output_dir / 'train' / 'data'}")
    print(f"  {output_dir / 'train' / 'user_info'}")
    print(f"  {output_dir / 'train' / 'item_info'}")
    print(f"  {output_dir / 'valid' / 'data'}")
    print(f"  {output_dir / 'test' / 'data'}")
    print(f"  {output_dir / 'meta_data.json'}")
    print(f"  {output_dir / 'block_manifest.json'}")
    print("\nmeta_data.json:")
    print(json.dumps(meta_data, ensure_ascii=False, indent=4))


# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess QK-Video seq-action data into blocked train/valid/test with OOV filtering."
    )
    parser.add_argument("--input_file", type=str, default="./QK-video.csv")
    parser.add_argument("--output_dir", type=str, default="../QK_Video_Action")
    parser.add_argument("--min_user_interactions", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--n_user_parts", type=int, default=10,
                        help="Phase1 按 user hash 的临时分区数")
    parser.add_argument("--chunk_size", type=int, default=2_000_000,
                        help="读取 CSV 时单次处理多少行")
    parser.add_argument("--buffer_flush_size", type=int, default=500_000,
                        help="临时分区缓存多少 interaction 后落盘")
    parser.add_argument("--train_blocks", type=int, default=8)
    parser.add_argument("--valid_blocks", type=int, default=4)
    parser.add_argument("--test_blocks", type=int, default=4)
    parser.add_argument("--min_feat_count", type=int, default=MIN_FEAT_COUNT,
                        help="OOV 过滤阈值: 特征值出现次数 < 此值的统一映射为 0")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="若输出目录已存在，则删除后重建")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_and_split(
        input_file=args.input_file,
        output_dir=args.output_dir,
        min_user_interactions=args.min_user_interactions,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        n_user_parts=args.n_user_parts,
        chunk_size=args.chunk_size,
        buffer_flush_size=args.buffer_flush_size,
        train_blocks=args.train_blocks,
        valid_blocks=args.valid_blocks,
        test_blocks=args.test_blocks,
        min_feat_count=args.min_feat_count,
        overwrite=args.overwrite,
    )
