#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_QK_seq_action.py — QK-Video memory-efficient blocked preprocessing
================================================================================================
Use user-hash partitions and blocked output, following the KuaiRand and TAAC2025 pipelines.

Split strategy (8:1:1 per user):
  - Preserve each user's released event order.
  - Use the earliest ~80% for training, the next ~10% for validation, and the latest ~10% for testing.

Processing flow:
  Phase 1 — CSV chunked read → clean → write to Parquet temporary files partitioned by user hash
             At the same time, collect global statistical information (number of user interactions, item category mapping, unique feature values, feature value frequency)
  Phase 2 — Partition-by-partition: Encoding features + segmentation according to user sequence ratio + direct writing of block data/user_info
  Phase 3 — Build item_info for each block + save meta_data/block_manifest + clean temporary files

New features:
  1. Storage in blocks (consistent with TAAC2025):
     - Each split is internally divided into multiple blocks
     - Each block generates its own data / user_info / item_info independently
     - block internal user_index / item_index remap to local continuous id
     - user_id / item_id maintains the global id and does not affect embedding consistency
  2. OOV low-frequency feature filtering (min_feat_count=2):
     - Count the occurrences of each feature value during the vocab construction phase
     - Feature values occurring fewer than min_feat_count times are uniformly mapped to 0 (unknown/padding)
     - Only works on non-ID category features (video_category, watching_times, gender, age)
     - Does not affect global ID mappings such as user_id / item_id / action

Dependencies:
    pip install pandas numpy pyarrow

usage:
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
#  1. Constants and column definitions
# ================================================================

# OOV filtering threshold: feature values occurring fewer than this threshold are mapped to 0 (unknown/padding)
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

# Features that require OOV filtering (non-ID category features)
OOV_FILTER_FEATURES = ["video_category", "watching_times", "gender", "age"]

# Item features stored in item_info
ITEM_STATIC_FEATURES = ["video_category"]

# Precomputed action lookup table: 4 binary labels, a total of 2^4 = 16 combinations
_ACTION_LOOKUP = np.empty(16, dtype=object)
for _k in range(16):
    if _k == 0:
        _ACTION_LOOKUP[_k] = "exposure"
    else:
        _ACTION_LOOKUP[_k] = "|".join(
            col for bit, col in zip([8, 4, 2, 1], LABEL_COLUMNS) if _k & bit
        )


# ================================================================
#  2. Utilities
# ================================================================

def check_required_columns(columns):
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"The following column is missing from the dataset: {missing}")


def build_ordered_vocab(values, start=1):
    """Constructs a vocab from an iterable object, encoding starting at start and 0 reserved for padding/unknown."""
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
    """Proportionately distribute a single user behavior sequence of length n to train/valid/test, with at least 1 each."""
    if n < 3:
        raise ValueError(f"The number of single user actions must be >= 3, currently n={n}")
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
                    f"Unable to assign at least 1 sample to each split: n={n}, counts={counts.tolist()}"
                )
            donor = donors[np.argmax(counts[donors])]
            counts[donor] -= 1
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def check_static_consistency(df, key_col, value_cols, name):
    """Check whether the static features are consistent, and issue a warning if they are inconsistent."""
    inconsistent = {}
    for col in value_cols:
        nunique = df.groupby(key_col, sort=False)[col].nunique(dropna=False)
        bad = int((nunique > 1).sum())
        if bad > 0:
            inconsistent[col] = bad
    if inconsistent:
        warnings.warn(
            f"{name} There is a situation where the same key corresponds to multiple values."
            f"The last occurrence of the value will be retained: {inconsistent}"
        )


# ================================================================
#  3. Phase 1: CSV → Partitioned Parquet + Global Statistics
# ================================================================

def _clean_chunk(chunk: pd.DataFrame, global_row_offset: int):
    """
    Perform basic cleaning on a single chunk:
      1. Keep only necessary columns
      2. label → float32
      3. categorical → clean str
      4. Discard user_id / item_id missing rows
      5. Construct exposure + action (vector lookup table)
      6. Assign global _row_id
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

    # Discard missing user_id / item_id rows
    mask = (chunk["user_id"] != "__MISSING__") & (chunk["item_id"] != "__MISSING__")
    chunk = chunk[mask].reset_index(drop=True)

    if len(chunk) == 0:
        return chunk, global_row_offset

    # Construct exposure + action (vectorized lookup table, instead of row-by-row apply)
    label_vals = np.column_stack([chunk[c].values for c in LABEL_COLUMNS])
    binary = (label_vals > 0).astype(np.uint8)
    chunk["exposure"] = (binary.sum(axis=1) == 0).astype(np.float32)
    keys = binary[:, 0] * 8 + binary[:, 1] * 4 + binary[:, 2] * 2 + binary[:, 3]
    chunk["action"] = _ACTION_LOOKUP[keys]
    del label_vals, binary, keys

    # Global _row_id (preserves original row order across chunks)
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
    Read CSV chunk by chunk → clean → write to Parquet partitioned by user_id hash.
    Also collected:
      - user_counts: {user_id_str: interaction_count}
      - item_category_codes: {item_id_str: video_category_code}
      - unique_values: {feature_name: [unique values in first-seen order]}
      - feat_value_counters: {feature_name: {value_str: count}}
        Used for subsequent OOV low-frequency feature filtering
    """
    print(
        f"\n[Phase 1] CSV → Partition Parquet"
        f"(n_parts={n_parts}, chunk_size={chunk_size:,})"
    )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_dir / f"part_{p:03d}").mkdir(exist_ok=True)

    # Check if the columns are complete
    sample = pd.read_csv(input_file, nrows=5)
    check_required_columns(sample.columns)
    del sample

    # ---- Global Statistics Container ----
    user_counts = defaultdict(int)
    item_category_codes = {}

    # Track unique values in order of first occurrence
    unique_trackers = {
        "video_category": {},
        "watching_times": {},
        "gender": {},
        "age": {},
        "action": {},
    }

    # ---- Feature value frequency statistics container ----
    feat_value_counters = {
        "video_category": defaultdict(int),
        "watching_times": defaultdict(int),
        "gender": defaultdict(int),
        "age": defaultdict(int),
    }

    # ---- Partition write buffer ----
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

        # ---- Collect statistics ----
        vc = chunk["user_id"].value_counts(sort=False)
        for uid, cnt in zip(vc.index, vc.values):
            user_counts[uid] += int(cnt)
        for feat_name in unique_trackers:
            for v in chunk[feat_name].unique():
                tracker = unique_trackers[feat_name]
                tracker.setdefault(v, len(tracker))

        # video_category is an item static feature. Save using compact category numbers
        # Mapping of item_id -> category; repeated items retain the last occurrence of the value.
        category_codes = chunk["video_category"].map(
            unique_trackers["video_category"]
        ).to_numpy(dtype=np.int16, copy=False)
        item_category_codes.update(zip(chunk["item_id"], category_codes))

        # ---- Statistics OOV_FILTER_FEATURES frequency ----
        for feat_name in feat_value_counters:
            fvc = chunk[feat_name].value_counts()
            for v, cnt in fvc.items():
                feat_value_counters[feat_name][v] += int(cnt)

        # ---- Hash partition routing by user_id ----
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
            print(f"    chunk {chunk_idx + 1}: accumulated {total_rows:,} rows")

    # ---- Final flash ----
    for pid in range(n_parts):
        flush(pid)

    del buffers, buf_sizes
    gc.collect()

    unique_values = {k: list(v.keys()) for k, v in unique_trackers.items()}

    # Convert to plain dict
    feat_value_counters_plain = {
        k: dict(v) for k, v in feat_value_counters.items()
    }

    print(
        f"  Done: {total_rows:,} row,"
        f"{len(user_counts):,} User, {len(item_category_codes):,} Item\n"
    )
    return dict(user_counts), item_category_codes, unique_values, feat_value_counters_plain


# ================================================================
#  4. Vocab construction (including OOV filtering)
# ================================================================

def build_vocabs_with_oov_filter(
    unique_values: dict,
    feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """Build a vocab based on frequency statistics and filter feature values occurring fewer than min_feat_count times.

    Args:
        unique_values: {feature_name: [value_str, ...]} in order of first occurrence
        feat_value_counters: {feature_name: {value_str: count}}
        min_feat_count: frequency threshold

    Returns:
        vocabs: {feature_name: {value_str: int_id}}
        - OOV_FILTER_FEATURES will filter based on frequency
        - action is not filtered (because action comes from label combination)
    """
    vocabs = {}

    for feat_name, uv in unique_values.items():
        if feat_name in OOV_FILTER_FEATURES and feat_name in feat_value_counters:
            # Build vocab based on OOV filtering
            counter = feat_value_counters[feat_name]
            vocab = {}
            idx = 1
            # __MISSING__ always reserved
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
            # No filtering (like action)
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
    To process a user partition:
      1. Read all parquet files under the partition
      2. Filter valid users
      3. Sort by (user_id, _row_id) → keep original behavior order
      4. Encoding user_index / item_index / categorical features / action
      5. Split train/valid/test according to user sequence ratio
      6. Build user_info fragment

    Returns dict{train, valid, test, user_info} or None.
    """
    part_dir = tmp_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None

    # ---- Read + Filter ----
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

    # ---- Sorting: Keep the original behavior order by _row_id within the user ----
    df.sort_values(["user_id", "_row_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- User/item static feature consistency check ----
    check_static_consistency(
        df, key_col="user_id", value_cols=["gender", "age"],
        name=f"user side (partition {pid})",
    )
    check_static_consistency(
        df, key_col="item_id", value_cols=["video_category"],
        name=f"item side (partition {pid})",
    )

    # ---- Encoding ID ----
    df["user_index"] = df["user_id"].map(user_idx_map).astype(np.int32)
    df["item_index"] = df["item_id"].map(item_idx_map).fillna(0).astype(np.int32)

    # user_id categorical id: 1-based
    df["user_id"] = (df["user_index"] + 1).astype(np.int32)

    # ---- Encoding categorical features (including OOV filtering) ----
    for col in ["video_category", "watching_times", "gender", "age"]:
        df[col] = df[col].map(vocabs[col]).fillna(0).astype(np.int32)
    df["action"] = df["action"].map(vocabs["action"]).fillna(0).astype(np.int32)

    # ---- seq_len: How many historical behaviors have preceded the current behavior ----
    df["seq_len"] = df.groupby("user_index", sort=False).cumcount().astype(np.int32)

    # ---- Build user_info fragment ----
    user_info_rows = []
    for uidx, gdf in df.groupby("user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(uidx),
                "full_item_seq": gdf["item_index"].tolist(),
                "full_action_seq": gdf["action"].tolist(),
            }
        )

    # ---- Split according to user sequence ratio ----
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
#  6. Split Block Manager (refer to TAAC2025 implementation)
# ================================================================

class SplitBlockManager:
    """
    Divide a split (train/valid/test) into multiple blocks and write each block separately:
      - data/part-xxxxx.parquet
      - user_info/part-xxxxx.parquet
      - item_info/part-xxxxx.parquet

    inside block:
      - user_index / item_index remapped to block-local dense index
      - user_id / item_id maintains the global id and does not affect embedding consistency
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
        """Append the split data of a partition to a block."""
        if len(split_df) == 0:
            return None

        bid = self.choose_block(len(split_df))
        self.partition_to_block[int(pid)] = int(bid)

        # Build a lookup table of user_index -> user_info_row
        ui_lookup = {r["user_index"]: r for r in user_info_rows}

        active_users = pd.unique(split_df["user_index"].astype(np.int64))
        active_users_set = set(active_users.tolist())

        out_ui_rows = []
        for g_user in active_users_set:
            if g_user not in ui_lookup:
                raise ValueError(
                    f"[{self.split_name}] partition={pid} user_index={g_user} "
                    f"Does not exist in user_info, logical exception."
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
        """Generate item_info.parquet for each block.

        Args:
            global_item_lookup: DataFrame indexed by global item_index,
                                Contains the ITEM_STATIC_FEATURES column
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
    item_category_codes: dict,
    category_values: list,
    vocabs: dict,
) -> pd.DataFrame:
    """Build a true video_category lookup indexed by global item_index."""
    num_items = max(item_idx_map.values()) if item_idx_map else 0
    category_vocab = vocabs["video_category"]
    encoded_by_code = np.fromiter(
        (category_vocab.get(value, 0) for value in category_values),
        dtype=np.int32,
        count=len(category_values),
    )
    global_item_indices = np.fromiter(
        (item_idx_map[item_id] for item_id in item_category_codes),
        dtype=np.int32,
        count=len(item_category_codes),
    )
    category_codes = np.fromiter(
        item_category_codes.values(),
        dtype=np.int32,
        count=len(item_category_codes),
    )

    encoded_categories = np.zeros(num_items + 1, dtype=np.int32)
    encoded_categories[global_item_indices] = encoded_by_code[category_codes]

    df = pd.DataFrame(
        {"video_category": encoded_categories},
        index=np.arange(num_items + 1, dtype=np.int32),
    )
    df.index.name = "global_item_index"
    return df


# ================================================================
#  7. Main process
# ================================================================

def preprocess_and_split(
    input_file="QK-video.csv",
    output_dir="./data/QK_Video",
    min_user_interactions=3,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    n_user_parts=20,
    chunk_size=4_000_000,
    buffer_flush_size=1_000_000,
    train_blocks=32,
    valid_blocks=8,
    test_blocks=8,
    min_feat_count=MIN_FEAT_COUNT,
    overwrite=False,
):
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"train_ratio + valid_ratio + test_ratio must equal 1, currently {ratio_sum}"
        )

    input_file_path = Path(input_file).resolve()
    output_dir = Path(output_dir).resolve()

    block_counts = (int(train_blocks), int(valid_blocks), int(test_blocks))
    if min(block_counts) <= 0:
        raise ValueError("train_blocks / valid_blocks / test_blocks must be positive integers.")
    target_blocks = max(block_counts)
    if n_user_parts < target_blocks:
        print(
            f"[Config] n_user_parts={n_user_parts} is less than the maximum target block number {target_blocks},"
            f"Automatically adjust to {target_blocks} to prevent all target blocks from being generated."
        )
        n_user_parts = target_blocks

    # Security check: prohibit output_dir from being the directory where input_file is located or its ancestor.
    # Otherwise, overwrite will also delete the original data.
    input_parent = input_file_path.parent
    if input_parent == output_dir or input_parent.is_relative_to(output_dir):
        raise ValueError(
            f"output_dir cannot be equal to or contain the directory where the original data is located, otherwise the original data will be deleted! \n"
            f"  input_file:  {input_file_path}\n"
            f"  data_parent: {input_parent}\n"
            f"  output_dir:  {output_dir}\n"
            f"Please set output_dir to a different directory than the original data."
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
    #  Step 1/8: Phase 1 — CSV → Partition Parquet + Global Statistics
    # ================================================================
    user_counts, item_category_codes, unique_values, feat_value_counters = phase1_partition_to_parquet(
        input_file=input_file,
        tmp_dir=tmp_dir,
        n_parts=n_user_parts,
        chunk_size=chunk_size,
        buffer_flush_size=buffer_flush_size,
    )
    gc.collect()

    # ================================================================
    #  Step 2/8: Filter low-frequency users + build global ID mapping + Vocab (including OOV filtering)
    # ================================================================
    print("[Step 2/8] Filter low-frequency users + build global mapping + Vocab (including OOV filtering)")

    valid_users = {
        u for u, c in user_counts.items() if c >= min_user_interactions
    }
    n_dropped = len(user_counts) - len(valid_users)
    dropped_rows = sum(
        c for u, c in user_counts.items() if c < min_user_interactions
    )
    if n_dropped > 0:
        print(
            f"  [Info] Filter users with interaction count < {min_user_interactions}:"
            f"users={n_dropped:,}, rows={dropped_rows:,}"
        )
    print(f"  Valid user: {len(valid_users):,}")

    if not valid_users:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("The data is empty after filtering, please check the original data or reduce min_user_interactions.")

    # user_idx_map: 0-based (sorted for determinism)
    sorted_users = sorted(valid_users)
    user_idx_map = {u: i for i, u in enumerate(sorted_users)}

    # item_idx_map: 1-based, 0 = padding
    sorted_items = sorted(item_category_codes)
    item_idx_map = {it: i + 1 for i, it in enumerate(sorted_items)}
    print(f"  Number of items: {len(item_idx_map):,}")

    # Build vocab (1-based, 0=padding/unknown), including OOV filtering
    print(f"  [OOV Filter] min_feat_count={min_feat_count}")
    vocabs = build_vocabs_with_oov_filter(
        unique_values, feat_value_counters, min_feat_count=min_feat_count
    )

    global_item_lookup = build_global_item_lookup(
        item_idx_map=item_idx_map,
        item_category_codes=item_category_codes,
        category_values=unique_values["video_category"],
        vocabs=vocabs,
    )

    # vocab_size (including padding bit)
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
    print(f"  action type: {len(vocabs['action'])}")

    del user_counts, item_category_codes, unique_values, feat_value_counters, sorted_users, sorted_items
    gc.collect()

    # ================================================================
    #  Step 3/8: Phase 2 — Partition-by-partition encoding + segmentation + writing block data/user_info
    # ================================================================
    print(f"\n[Step 3/8] Encode partition by partition + split according to user sequence proportion + write block data/user_info")

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
        print(f"  partition {pid + 1:3d}/{n_user_parts}: accumulated {done:,} rows")

        del result
        gc.collect()

    # Close writers
    for mgr in managers.values():
        mgr.close_writers()

    total = sum(sample_counts.values())
    print(
        f"\n  train={sample_counts['train']:,}  "
        f"valid={sample_counts['valid']:,}  "
        f"test={sample_counts['test']:,}"
    )
    print(f"  TOTAL={total:,}")

    if total == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("The data is empty after processing, please check the original data or adjust the parameters.")
    for s in ["train", "valid", "test"]:
        if sample_counts[s] == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ValueError(f"The {s} set is empty, please check the data or adjust the segmentation parameters.")

    print(
        f"  Segmentation ratio (target): train={train_ratio:.2f},"
        f"valid={valid_ratio:.2f}, test={test_ratio:.2f}"
    )

    del valid_users, user_idx_map, item_idx_map
    gc.collect()

    # ================================================================
    #  Step 4/8: Construct item_info for each block
    # ================================================================
    print(f"\n[Step 4/8] Build item_info for each block")

    for split_name in ["train", "valid", "test"]:
        print(f"  [Build item_info] {split_name}")
        managers[split_name].write_item_info_blocks(global_item_lookup)
        gc.collect()

    del global_item_lookup
    gc.collect()

    # ================================================================
    #  Step 5/8: Save meta_data + block_manifest
    # ================================================================
    print(f"\n[Step 5/8] Save meta_data + block_manifest")

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
            "rule": "The uniform mapping of feature value occurrences < min_feat_count is 0 (unknown/padding)",
        },
        "blocked_layout": {
            "n_user_parts": int(n_user_parts),
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
                "Under the same split, data/user_info/item_info uses the same part-xxxxx number for paired reading."
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
            "Encoded action vocabulary for dataloader based on full_action_seq"
            "Construct task-specific token masks."
        ),
        "user_info_schema": {
            "fields": [
                "user_index",
                "full_item_seq",
                "full_action_seq",
            ],
            "desc": (
                "The item index in user_index / full_item_seq here is all block-local index;"
                "full_action_seq is the global time sequence sequence."
            ),
        },
        "item_info_schema": {
            "fields": [
                "item_index",
                "item_id",
            ] + ITEM_STATIC_FEATURES,
            "desc": (
                "item_index is block-local index; item_id is global item feature id,"
                "Used for embedding consistency."
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
    #  Step 6/8: Clean up temporary files
    # ================================================================
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  Temporary partition files have been cleaned")

    # ================================================================
    #  Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  QK-Video Preprocess Done (blocked partition processing + blocked output + OOV filtering)")
    print("=" * 70)
    print(f"Output directory: {output_dir}\n")
    print("Directory structure example:")
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
                        help="Phase1 Number of temporary partitions by user hash")
    parser.add_argument("--chunk_size", type=int, default=4_000_000,
                        help="How many rows to process at a time when reading CSV")
    parser.add_argument("--buffer_flush_size", type=int, default=1_000_000,
                        help="How much interaction is cached in the temporary partition and then dropped to disk?")
    parser.add_argument("--train_blocks", type=int, default=32)
    parser.add_argument("--valid_blocks", type=int, default=8)
    parser.add_argument("--test_blocks", type=int, default=8)
    parser.add_argument("--min_feat_count", type=int, default=MIN_FEAT_COUNT,
                        help="OOV filtering threshold: The number of occurrences of the feature value < the unified mapping of this value is 0")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="If the output directory already exists, delete it and recreate it.")
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
