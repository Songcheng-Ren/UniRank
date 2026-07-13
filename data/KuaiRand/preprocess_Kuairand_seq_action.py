#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_Kuairand_seq_action.py — KuaiRand-27K (内存优化版, 按日期比例切分, blocked 输出)
================================================================================================
通过分批处理避免内存溢出 (OOM)，并采用与 TAAC2025 一致的分 block 存储进一步降低内存峰值。

切分策略（按日期 8:1:1）:
  - 将所有日期排序后，按天数比例 8:1:1 划分
  - 前 ~80% 天 → 训练集
  - 中间 ~10% 天 → 验证集
  - 最后 ~10% 天 → 测试集

优化策略:
  Phase 1 — CSV → 按用户哈希分区的 Parquet 中间文件
  Phase 2 — 逐分区编码 + 按日期切分 + 直接写 block data/user_info
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
     - 仅作用于非 ID 类类别特征 (USER_STATIC_FEATURES / ITEM_STATIC_FEATURES)
     - 不影响 user_id / item_id / action 等全局 ID 映射

本版本说明:
  - 不再构建 user_info.behavior_type_mask
  - 改为在 meta_data.json 中保存 action_vocab
  - 新增 user_info.full_timestamp_seq
  - 供 dataloader 在训练时基于 full_action_seq 构造 task-specific token masks

依赖:
    pip install pandas numpy pyarrow
"""

import argparse
import gc
import json
import shutil
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

LOG_FILES = [
    "log_random_4_22_to_5_08_27k.csv",
    "log_standard_4_08_to_4_21_27k_part1.csv",
    "log_standard_4_08_to_4_21_27k_part2.csv",
    "log_standard_4_22_to_5_08_27k_part1.csv",
    "log_standard_4_22_to_5_08_27k_part2.csv",
]

LOG_LOAD_COLUMNS = [
    "user_id", "video_id", "time_ms",
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view",
    "play_time_ms", "tab",
]

LOG_DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "time_ms": "int64",
    "is_click": "int8",
    "is_like": "int8",
    "is_follow": "int8",
    "is_comment": "int8",
    "is_forward": "int8",
    "long_view": "int8",
    "play_time_ms": "int32",
    "tab": "int8",
}

LABEL_COLUMNS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view",
]

USER_STATIC_FEATURES = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]

CONTEXT_FEATURES = ["tab", "day_of_week", "is_weekend", "hour"]

ITEM_STATIC_FEATURES = ["video_type", "primary_tag", "music_type", "duration_bucket"]

FINAL_COLUMNS = (
    ["user_index", "item_index", "seq_len", "user_id"]
    + USER_STATIC_FEATURES + CONTEXT_FEATURES + LABEL_COLUMNS
)


# ================================================================
#  2. 工具函数
# ================================================================

def build_ordered_vocab(values, start=1):
    """从可迭代对象构建 vocab，start=1 时 0 预留给 padding/unknown。"""
    uniq = list(dict.fromkeys(str(v) for v in values))
    return {v: i for i, v in enumerate(uniq, start=start)}


PLAY_TIME_BINS = [-1, 0, 1000, 3000, 7000, 18000, 60000, float("inf")]
PLAY_TIME_LABELS = ["0", "0-1s", "1-3s", "3-7s", "7-18s", "18-60s", "60s+"]

DURATION_BINS = [-1, 0, 3000, 7000, 15000, 30000, 60000, float("inf")]
DURATION_LABELS = ["0", "0-3s", "3-7s", "7-15s", "15-30s", "30-60s", "60s+"]


def bucket_play_time(s: pd.Series) -> pd.Series:
    return pd.cut(s.fillna(0), bins=PLAY_TIME_BINS, labels=PLAY_TIME_LABELS).astype(str)


def bucket_duration(s: pd.Series) -> pd.Series:
    return pd.cut(s.fillna(0), bins=DURATION_BINS, labels=DURATION_LABELS).astype(str)


def fmt_date_int(d: int) -> str:
    return f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"


def build_date_split(sorted_dates, train_ratio, valid_ratio, test_ratio):
    n_days = len(sorted_dates)
    if n_days < 3:
        raise ValueError(
            f"数据中仅有 {n_days} 个不同日期，至少需要 3 天才能按日期切分 train/valid/test。"
        )

    ratios = np.array([train_ratio, valid_ratio, test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()

    raw = ratios * n_days
    counts = np.floor(raw).astype(int)
    rem = n_days - counts.sum()

    if rem > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for i in range(rem):
            counts[order[i % 3]] += 1

    for idx in range(3):
        while counts[idx] == 0:
            donors = np.where(counts > 1)[0]
            if len(donors) == 0:
                break
            donor = donors[np.argmax(counts[donors])]
            counts[donor] -= 1
            counts[idx] += 1

    n_train, n_valid, n_test = int(counts[0]), int(counts[1]), int(counts[2])

    train_dates = sorted_dates[:n_train]
    valid_dates = sorted_dates[n_train:n_train + n_valid]
    test_dates = sorted_dates[n_train + n_valid:]

    return {
        "n_days": n_days,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "train_dates": train_dates,
        "valid_dates": valid_dates,
        "test_dates": test_dates,
        "valid_start_date": valid_dates[0],
        "test_start_date": test_dates[0],
    }


# ================================================================
#  3. Vocab & 映射构建（不依赖日志扫描）
# ================================================================

def load_user_features(data_dir: Path) -> pd.DataFrame:
    """加载用户特征（27K 行，很小）。"""
    fp = data_dir / "user_features_27k.csv"
    print(f"  [Load] {fp.name}")
    cols = ["user_id"] + USER_STATIC_FEATURES
    uf = pd.read_csv(fp, usecols=cols)
    uf["user_id"] = uf["user_id"].astype("int32")
    for col in USER_STATIC_FEATURES:
        uf[col] = uf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
    print(f"         {len(uf):,} users")
    return uf


def build_all_vocabs(uf: pd.DataFrame) -> dict:
    """从用户特征表 + 已知域构建所有类别特征 vocab（0=padding/unknown）。
    注意: 此处只构建静态 vocab，不基于频次过滤。
    OOV 过滤在 build_vocabs_with_oov_filter 中完成。"""
    vocabs = {}
    for col in USER_STATIC_FEATURES:
        vals = sorted(set(uf[col].unique()) | {"__MISSING__"})
        vocabs[col] = {v: i + 1 for i, v in enumerate(vals)}
    vocabs["tab"] = {str(i): i + 1 for i in range(15)}
    vocabs["day_of_week"] = {str(i): i + 1 for i in range(7)}
    vocabs["is_weekend"] = {"0": 1, "1": 2}
    vocabs["hour"] = {str(i): i + 1 for i in range(24)}
    vocabs["play_time_bucket"] = {l: i + 1 for i, l in enumerate(PLAY_TIME_LABELS)}
    return vocabs


def build_vocabs_with_oov_filter(
    uf: pd.DataFrame,
    feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """基于频次统计构建 vocab，过滤出现次数 < min_feat_count 的特征值。

    Args:
        uf: 用户特征 DataFrame (USER_STATIC_FEATURES 为字符串形式)
        feat_value_counters: {feature_name: {value_int: count}}
            uf_enc 中编码后的 int 值频次
        min_feat_count: 频次阈值，低于此值的特征值不加入 vocab

    Returns:
        vocabs: {feature_name: {value_str: int_id}}
        - ID 列 (user_id, item_id, action) 不在此处理
        - USER_STATIC_FEATURES / ITEM_STATIC_FEATURES 会基于频次过滤
        - 其他固定域 (tab, day_of_week, is_weekend, hour, play_time_bucket) 不过滤

    说明:
        由于 uf_enc 是基于初始 vocab 编码的 int16，feat_value_counters 中的 key 也是
        这个编码后的 int 值。我们使用这个 int 值作为 vocab 的 key，而不是字符串。
        最终输出 vocab 为 {int_value_str: int_id} 形式。
    """
    vocabs = {}

    # ---- USER_STATIC_FEATURES: 基于频次过滤 ----
    for col in USER_STATIC_FEATURES:
        vocab = {}
        idx = 1
        # uf_enc 中已编码为 int16，但此处我们直接用 int 值作为 key
        # 收集所有出现过的 int 值 (包括 0 表示 __MISSING__)
        if col in feat_value_counters:
            counter = feat_value_counters[col]
            # 0 表示 __MISSING__，始终保留
            all_int_vals = set(counter.keys()) | {0}
            for v in sorted(all_int_vals):
                cnt = int(counter.get(v, 0))
                # 0 (MISSING) 始终保留；其他值频次 < min_feat_count 则过滤
                if v != 0 and cnt > 0 and cnt < min_feat_count:
                    continue
                vocab[v] = idx
                idx += 1
        else:
            # 没有频次统计的特征，全部保留
            vocab[0] = 1
            idx = 2
        vocabs[col] = vocab

    # ---- 固定域: 不过滤 ----
    vocabs["tab"] = {str(i): i + 1 for i in range(15)}
    vocabs["day_of_week"] = {str(i): i + 1 for i in range(7)}
    vocabs["is_weekend"] = {"0": 1, "1": 2}
    vocabs["hour"] = {str(i): i + 1 for i in range(24)}
    vocabs["play_time_bucket"] = {l: i + 1 for i, l in enumerate(PLAY_TIME_LABELS)}

    return vocabs


def build_item_vocabs_with_oov_filter(
    item_feat_df: pd.DataFrame,
    feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """构建 item 特征 vocab，基于频次过滤低频值。

    Args:
        item_feat_df: item 特征 DataFrame (已清洗为 str)
        feat_value_counters: {feature_name: {value_str: count}}
        min_feat_count: 频次阈值

    Returns:
        vocabs: {feature_name: {value_str: int_id}}
    """
    vocabs = {}
    for col in ITEM_STATIC_FEATURES:
        vocab = {}
        idx = 1
        all_vals = set(item_feat_df[col].unique()) | {"__MISSING__"}
        if col in feat_value_counters:
            counter = feat_value_counters[col]
            for v in sorted(all_vals):
                cnt = int(counter.get(v, 0))
                if cnt > 0 and cnt < min_feat_count:
                    continue
                vocab[v] = idx
                idx += 1
        else:
            for v in sorted(all_vals):
                vocab[v] = idx
                idx += 1
        vocabs[col] = vocab
    return vocabs


def build_action_maps():
    """枚举所有 2^6=64 种 action pattern → (pattern_int→name, name→code)。"""
    pat2name = {}
    for p in range(64):
        if p == 0:
            pat2name[p] = "exposure"
        else:
            parts = [c for i, c in enumerate(LABEL_COLUMNS) if p & (1 << i)]
            pat2name[p] = "|".join(parts)
    name2code = {n: i + 1 for i, n in enumerate(sorted(set(pat2name.values())))}
    return pat2name, name2code


def encode_user_features_to_int(uf: pd.DataFrame, vocabs: dict) -> pd.DataFrame:
    """将用户特征预编码为 int16，合并到日志时大幅节省内存。"""
    uf_enc = uf[["user_id"]].copy()
    for col in USER_STATIC_FEATURES:
        uf_enc[col] = uf[col].map(vocabs[col]).fillna(0).astype("int16")
    return uf_enc


# ================================================================
#  4. Phase 1: CSV → Partitioned Parquet
# ================================================================

def _preprocess_chunk(
    chunk: pd.DataFrame,
    uf_enc: pd.DataFrame,
    vocabs: dict,
):
    """
    对单个 chunk (~2M 行) 执行:
      1. 时间戳分解 (保留 time_ms 供后续排序和 user_info.full_timestamp_seq)
      2. 提取 date 列 (int32, YYYYMMDD) 供按日期切分
      3. play_time 分桶 + 上下文特征编码 → int8
      4. 合并预编码用户特征 → int16

    注意: 此函数在 Phase 1 和 Phase 2 之间共享。Phase 1 调用时
    vocabs 可能为空字典（此时会先保留 str 特征值），Phase 2 时
    vocabs 已构造完成。

    但为简化实现，Phase 1 调用此函数时只做时间戳和上下文编码，
    不做 USER_STATIC_FEATURES 编码（因为最终 vocab 取决于 OOV 过滤）。
    """
    # ---- 时间戳分解 ----
    dt = pd.to_datetime(chunk["time_ms"], unit="ms", utc=True).dt.tz_convert(
        "Asia/Shanghai"
    )
    chunk["day_of_week"] = dt.dt.dayofweek.astype("int8")
    chunk["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    chunk["hour"] = dt.dt.hour.astype("int8")

    # ---- 提取日期列 (YYYYMMDD int32) ----
    chunk["date"] = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).astype("int32")
    del dt

    # ---- play_time 分桶 + 编码 ----
    pt_bucket_str = bucket_play_time(chunk["play_time_ms"])
    chunk["play_time_bucket"] = (
        pt_bucket_str.map(vocabs["play_time_bucket"]).fillna(0).astype("int8")
    )
    del pt_bucket_str

    # ---- 上下文特征编码 ----
    chunk["tab"] = (
        chunk["tab"].astype(str).map(vocabs["tab"]).fillna(0).astype("int8")
    )
    chunk["day_of_week"] = (
        chunk["day_of_week"].astype(str).map(vocabs["day_of_week"]).fillna(0).astype("int8")
    )
    chunk["is_weekend"] = (
        chunk["is_weekend"].astype(str).map(vocabs["is_weekend"]).fillna(0).astype("int8")
    )
    chunk["hour"] = (
        chunk["hour"].astype(str).map(vocabs["hour"]).fillna(0).astype("int8")
    )

    # ---- 丢弃不再需要的列 ----
    chunk.drop(columns=["play_time_ms"], inplace=True)

    # ---- 去掉 ID 缺失行 ----
    chunk.dropna(subset=["user_id", "video_id"], inplace=True)

    return chunk


def phase1_partition_to_parquet(
    data_dir: Path,
    tmp_dir: Path,
    uf_enc: pd.DataFrame,
    vocabs: dict,
    n_parts: int,
    chunk_size: int,
    buffer_flush_size: int,
):
    """
    逐文件逐 chunk 读取 CSV → 预处理 → 按 user_id 哈希分区写入 Parquet。
    同时统计:
      - user_counts / item_ids / all_dates
      - feat_value_counters: {feature_name: {value_int: count}}
        用于后续 OOV 低频特征过滤

    注意: 为支持 OOV 过滤，Phase 1 不在此处对 USER_STATIC_FEATURES 做最终编码，
    而是仅做时间/上下文特征编码并保留 user_id 用于后续 merge。
    USER_STATIC_FEATURES 的频次统计通过 user_id 与 uf_enc 的映射计算。
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in range(n_parts):
        (tmp_dir / f"part_{p:03d}").mkdir(exist_ok=True)

    user_counts = defaultdict(int)
    item_ids = set()
    all_dates = set()

    # ---- 特征值频次统计容器 ----
    # uf_enc 中 USER_STATIC_FEATURES 已是 int16 编码（基于初始 vocab）
    # 但频次应基于「该用户在日志中出现了多少次」，故用 user_counts * 用户特征值
    # 这里改为: 在 Phase 1 完成后用 user_counts 和 uf_enc 联合计算
    feat_value_counters = None  # 占位，将在主流程中填充

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

    total_rows = 0
    for fname in LOG_FILES:
        print(f"  [Phase1] {fname}")
        reader = pd.read_csv(
            data_dir / fname,
            usecols=LOG_LOAD_COLUMNS,
            dtype=LOG_DTYPES,
            chunksize=chunk_size,
        )
        for chunk in reader:
            chunk = _preprocess_chunk(chunk, uf_enc, vocabs)

            vc = chunk["user_id"].value_counts(sort=False)
            for uid, cnt in zip(vc.index, vc.values):
                user_counts[uid] += int(cnt)
            item_ids.update(chunk["video_id"].unique().tolist())
            all_dates.update(chunk["date"].unique().tolist())

            part_arr = chunk["user_id"].values.astype(np.int64) % n_parts
            for pid in range(n_parts):
                mask = part_arr == pid
                n_match = int(mask.sum())
                if n_match > 0:
                    buffers[pid].append(chunk.loc[mask].copy())
                    buf_sizes[pid] += n_match
                    if buf_sizes[pid] >= buffer_flush_size:
                        flush(pid)

            total_rows += len(chunk)
            del chunk, part_arr
            gc.collect()

        print(f"         累计 {total_rows:,} 行")

    for pid in range(n_parts):
        flush(pid)

    print(
        f"  [Phase1] 完成: {total_rows:,} 行, "
        f"{len(user_counts):,} 用户, {len(item_ids):,} 视频, "
        f"{len(all_dates)} 个不同日期\n"
    )
    return dict(user_counts), item_ids, all_dates, feat_value_counters


def compute_feat_value_counters(
    user_counts: dict,
    uf_enc: pd.DataFrame,
) -> dict:
    """基于 user_counts 和 uf_enc 计算 USER_STATIC_FEATURES 的特征值频次。

    每个用户在日志中出现的次数 × 用户特征值 = 该特征值在日志中的总出现次数。

    Args:
        user_counts: {user_id: interaction_count}
        uf_enc: 预编码后的用户特征 DataFrame (含 user_id 和 USER_STATIC_FEATURES)

    Returns:
        {feature_name: {value_int: count}}
    """
    print("  [Phase1.5] 基于用户交互次数计算 USER_STATIC_FEATURES 频次 (用于 OOV 过滤)")
    counters = {col: defaultdict(int) for col in USER_STATIC_FEATURES}

    # 把 user_counts 转成 DataFrame
    uc_df = pd.DataFrame(
        list(user_counts.items()), columns=["user_id", "count"]
    )
    # 与 uf_enc 做 merge
    uc_df = uc_df.merge(uf_enc, on="user_id", how="left")

    for col in USER_STATIC_FEATURES:
        # 按 feature value 聚合 count
        grouped = uc_df.groupby(col)["count"].sum()
        for v, cnt in grouped.items():
            counters[col][int(v)] += int(cnt)

    del uc_df
    return counters


# ================================================================
#  5. Phase 2: Per-partition Process + Incremental Write
# ================================================================

def process_partition(
    tmp_dir: Path,
    pid: int,
    valid_users: set,
    user_idx_map: dict,
    item_idx_map: dict,
    uf_enc: pd.DataFrame,
    vocabs: dict,
    pat2name: dict,
    name2code: dict,
    valid_start_date: int,
    test_start_date: int,
):
    """
    处理一个用户分区:
      1. 读取分区下所有 parquet 文件
      2. 过滤有效用户
      3. 合并预编码用户特征 → int16
      4. 按 (user_id, time_ms) 排序
      5. 编码 user_index / item_index / action / exposure
      6. 按日期切分 train/valid/test
      7. 构建 user_info 片段（保留 full_item_seq / full_action_seq / full_timestamp_seq）

    返回 dict{train, valid, test, user_info} 或 None。
    """
    part_dir = tmp_dir / f"part_{pid:03d}"
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        return None

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

    # ---- 合并预编码用户特征 (int16) ----
    df = df.merge(uf_enc, on="user_id", how="left")
    for col in USER_STATIC_FEATURES:
        df[col] = df[col].fillna(0).astype("int16")
        # 应用 OOV 过滤: 不在 vocab 中的值映射为 0
        df[col] = df[col].map(vocabs[col]).fillna(0).astype("int32")

    # ---- 排序: 用户内按时间 ----
    df.sort_values(["user_id", "time_ms"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- 编码 ID ----
    df["user_index"] = df["user_id"].map(user_idx_map).astype(np.int32)
    df["item_index"] = df["video_id"].map(item_idx_map).fillna(0).astype(np.int32)
    df["user_id"] = (df["user_index"] + 1).astype(np.int32)

    # ---- 标签 → float32 ----
    for col in LABEL_COLUMNS:
        df[col] = df[col].astype(np.float32)

    # ---- exposure + action ----
    binary = df[LABEL_COLUMNS].values.astype(np.int8)
    df["exposure"] = (binary.sum(axis=1) == 0).astype(np.float32)
    pattern = np.zeros(len(df), dtype=np.int32)
    for i in range(len(LABEL_COLUMNS)):
        pattern += binary[:, i].astype(np.int32) << i
    df["action"] = (
        pd.Series(pattern, index=df.index).map(pat2name).map(name2code).astype(np.int32)
    )
    del binary, pattern

    # ---- 特征转 int32 ----
    for col in USER_STATIC_FEATURES + CONTEXT_FEATURES:
        df[col] = df[col].astype(np.int32)

    # ---- time_ms 确保 int64 ----
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce").fillna(0).astype(np.int64)

    # ---- seq_len ----
    df["seq_len"] = df.groupby("user_index", sort=False).cumcount().astype(np.int32)

    # ---- 构建 user_info ----
    user_info_rows = []
    for uidx, gdf in df.groupby("user_index", sort=True):
        user_info_rows.append(
            {
                "user_index": int(uidx),
                "full_item_seq": gdf["item_index"].astype(int).tolist(),
                "full_action_seq": gdf["action"].astype(int).tolist(),
                "full_timestamp_seq": gdf["time_ms"].astype(np.int64).tolist(),
            }
        )

    # ---- 按日期区间切分 ----
    date_col = df["date"]
    train_df = df[date_col < valid_start_date]
    valid_df = df[(date_col >= valid_start_date) & (date_col < test_start_date)]
    test_df = df[date_col >= test_start_date]

    def _select_final(sdf):
        if len(sdf) == 0:
            return pd.DataFrame(columns=FINAL_COLUMNS)
        present = [c for c in FINAL_COLUMNS if c in sdf.columns]
        return sdf[present].reset_index(drop=True)

    result = {
        "train": _select_final(train_df),
        "valid": _select_final(valid_df),
        "test": _select_final(test_df),
        "user_info": user_info_rows,
    }

    del df, train_df, valid_df, test_df
    gc.collect()
    return result


# ================================================================
#  6. Split Block 管理器 (参考 TAAC2025 实现)
# ================================================================

def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


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
        """将一个 partition 的 split 数据追加到某个 block 中。

        Args:
            pid: partition id
            split_df: 该 split 的 DataFrame (含 user_index, item_index 等)
            user_info_rows: list of dict (来自 process_partition 的 user_info)
        """
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

            g_time_seq = row.get("full_timestamp_seq", [])
            if not isinstance(g_time_seq, (list, tuple, np.ndarray)):
                g_time_seq = []

            out_ui_rows.append(
                {
                    "user_index": np.int32(l_user),
                    "full_item_seq": [int(x) for x in l_item_seq],
                    "full_action_seq": [int(x) for x in g_action_seq],
                    "full_timestamp_seq": [int(x) for x in g_time_seq],
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
    item_feat_df: pd.DataFrame,
    item_idx_map: dict,
) -> pd.DataFrame:
    """构建 global item lookup 表 (indexed by global item_index)。

    Args:
        item_feat_df: item 特征 DataFrame (含 video_id 和 ITEM_STATIC_FEATURES)
        item_idx_map: {video_id: global_item_index}

    Returns:
        DataFrame indexed by global_item_index, 列为 ITEM_STATIC_FEATURES
    """
    vf = item_feat_df[item_feat_df["video_id"].isin(item_idx_map)].copy()
    vf["global_item_index"] = vf["video_id"].map(item_idx_map).astype(np.int32)
    vf = vf[["global_item_index"] + ITEM_STATIC_FEATURES].drop_duplicates(
        subset=["global_item_index"], keep="last"
    )
    vf = vf.set_index("global_item_index").sort_index()

    for col in ITEM_STATIC_FEATURES:
        vf[col] = pd.to_numeric(vf[col], errors="coerce").fillna(0).astype(np.int32)

    return vf

def load_video_basic_features(data_dir: Path) -> pd.DataFrame:
    """加载视频基础特征，派生 primary_tag + duration_bucket。"""
    fp = data_dir / "video_features_basic_27k.csv"
    print(f"  [Load] {fp.name}")
    cols = ["video_id", "video_type", "music_type", "tag", "video_duration"]
    vf = pd.read_csv(fp, usecols=cols, dtype={"video_id": "int32"})
    vf["primary_tag"] = vf["tag"].astype(str).str.split(",").str[0].str.strip()
    vf["primary_tag"] = vf["primary_tag"].replace(
        {"nan": "__MISSING__", "": "__MISSING__"}
    )
    vf["duration_bucket"] = bucket_duration(vf["video_duration"].fillna(0))
    for col in ["video_type", "music_type"]:
        vf[col] = vf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
    vf.drop(columns=["tag", "video_duration"], inplace=True)
    print(f"         {len(vf):,} videos")
    return vf


def build_item_info(data_dir: Path, item_idx_map: dict, output_dir: Path):
    """
    item_info.parquet (用于非 blocked 模式):
      - 从 video_features_basic_27k.csv 加载
      - 仅保留日志中出现的视频
      - 编码 ITEM_STATIC_FEATURES
      - 第 0 行为 padding

    在 blocked 模式下，本函数不会被调用，改用 SplitBlockManager.write_item_info_blocks。
    """
    vf = load_video_basic_features(data_dir)

    vf = vf[vf["video_id"].isin(item_idx_map)].copy()
    vf["item_index"] = vf["video_id"].map(item_idx_map).astype(np.int32)
    vf["item_id"] = vf["item_index"]
    vf.drop(columns=["video_id"], inplace=True)
    vf.drop_duplicates(subset=["item_index"], keep="last", inplace=True)

    item_vs = {}
    for col in ITEM_STATIC_FEATURES:
        vf[col] = vf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
        vocab = build_ordered_vocab(vf[col], start=1)
        vf[col] = vf[col].map(vocab).astype(np.int32)
        item_vs[col] = len(vocab) + 1

    num_items = max(item_idx_map.values())
    item_info = pd.DataFrame(
        {
            "item_index": np.arange(num_items + 1, dtype=np.int32),
            "item_id": np.zeros(num_items + 1, dtype=np.int32),
            **{c: np.zeros(num_items + 1, dtype=np.int32) for c in ITEM_STATIC_FEATURES},
        }
    )

    vf_idx = vf.set_index("item_index")
    mask = item_info["item_index"].isin(vf_idx.index)
    matched = item_info.loc[mask, "item_index"].values
    for col in ["item_id"] + ITEM_STATIC_FEATURES:
        item_info.loc[mask, col] = vf_idx.loc[matched, col].values
    item_info = item_info.astype(np.int32)

    item_info.to_parquet(
        output_dir / "item_info.parquet", index=False, engine="pyarrow"
    )
    print(f"  [Saved] item_info.parquet ({num_items + 1:,} items incl. padding)")
    del vf, vf_idx
    return item_vs


def build_item_vocabs_from_data(
    data_dir: Path,
    item_idx_map: dict,
    feat_value_counters: dict,
    min_feat_count: int = MIN_FEAT_COUNT,
) -> dict:
    """构建 item 特征 vocab，支持 OOV 过滤。

    Args:
        data_dir: 数据目录
        item_idx_map: {video_id: global_item_index}
        feat_value_counters: {feature_name: {value_str: count}}
        min_feat_count: 频次阈值

    Returns:
        {feature_name: {value_str: int_id}}
    """
    vf = load_video_basic_features(data_dir)
    vf = vf[vf["video_id"].isin(item_idx_map)].copy()

    vocabs = {}
    for col in ITEM_STATIC_FEATURES:
        vf[col] = vf[col].astype(str).replace({"nan": "__MISSING__", "": "__MISSING__"})
        vocab = {}
        idx = 1
        all_vals = set(vf[col].unique()) | {"__MISSING__"}
        if col in feat_value_counters:
            counter = feat_value_counters[col]
            for v in sorted(all_vals):
                cnt = int(counter.get(v, 0))
                if cnt > 0 and cnt < min_feat_count:
                    continue
                vocab[v] = idx
                idx += 1
        else:
            for v in sorted(all_vals):
                vocab[v] = idx
                idx += 1
        vocabs[col] = vocab

    return vocabs


# ================================================================
#  7. 主流程
# ================================================================

def preprocess_and_split(
    data_dir: str = "./KuaiRand-27K/data",
    output_dir: str = "./data/KuaiRand_Video_Action",
    min_user_interactions: int = 10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_user_parts: int = 50,
    chunk_size: int = 2_000_000,
    buffer_flush_size: int = 500_000,
    train_blocks: int = 8,
    valid_blocks: int = 4,
    test_blocks: int = 4,
    min_feat_count: int = MIN_FEAT_COUNT,
    overwrite: bool = False,
):
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # 安全检查：禁止 output_dir 等于 data_dir 或为 data_dir 的祖先目录，
    # 否则 overwrite 会连带删除原始数据。
    if data_dir == output_dir or data_dir.is_relative_to(output_dir):
        raise ValueError(
            f"output_dir 不能等于或包含 data_dir，否则会删除原始数据！\n"
            f"  data_dir:   {data_dir}\n"
            f"  output_dir: {output_dir}\n"
            f"请将 output_dir 设为与 data_dir 不同的目录。"
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
    #  Step 1/9: 加载用户特征 + 构建初始 vocab (用于 Phase 1 中间编码)
    # ================================================================
    print("\n[Step 1/9] 加载用户特征 & 构建 vocab")
    uf = load_user_features(data_dir)
    vocabs_initial = build_all_vocabs(uf)
    pat2name, name2code = build_action_maps()
    uf_enc = encode_user_features_to_int(uf, vocabs_initial)
    print(f"  action 种类: {len(name2code)}")
    print("  用户特征已预编码为 int16 (中间编码, 后续会做 OOV 过滤)\n")

    # ================================================================
    #  Step 2/9: Phase 1 — CSV → 分区 Parquet
    # ================================================================
    print("[Step 2/9] Phase 1: CSV → 分区 Parquet")
    user_counts, item_ids, all_dates, _ = phase1_partition_to_parquet(
        data_dir,
        tmp_dir,
        uf_enc,
        vocabs_initial,
        n_parts=n_user_parts,
        chunk_size=chunk_size,
        buffer_flush_size=buffer_flush_size,
    )

    # ================================================================
    #  Step 2.5/9: 基于用户交互次数计算特征值频次 (用于 OOV 过滤)
    # ================================================================
    print("[Step 2.5/9] 计算特征值频次 (用于 OOV 过滤)")
    feat_value_counters = compute_feat_value_counters(user_counts, uf_enc)
    for col in USER_STATIC_FEATURES:
        n_total = len(feat_value_counters[col])
        n_kept = sum(1 for v, c in feat_value_counters[col].items() if c >= min_feat_count or v == 0)
        print(f"  {col}: total_values={n_total}, kept_after_oov_filter={n_kept}")

    # ================================================================
    #  Step 3/9: 基于 OOV 过滤构建最终 vocab
    # ================================================================
    print("[Step 3/9] 基于 OOV 过滤构建最终 vocab")
    vocabs = build_vocabs_with_oov_filter(
        uf, feat_value_counters, min_feat_count=min_feat_count
    )
    del uf, uf_enc, vocabs_initial, feat_value_counters
    gc.collect()
    print(f"  OOV 过滤阈值 (min_feat_count): {min_feat_count}")
    print(f"  action 种类: {len(name2code)}\n")

    # ================================================================
    #  Step 4/9: 按天数比例确定切分日期
    # ================================================================
    print("[Step 4/9] 按日期比例确定切分日期")
    sorted_dates = sorted(all_dates)
    split_info = build_date_split(
        sorted_dates,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    train_dates = split_info["train_dates"]
    valid_dates = split_info["valid_dates"]
    test_dates = split_info["test_dates"]
    valid_start_date = split_info["valid_start_date"]
    test_start_date = split_info["test_start_date"]

    print(f"  总天数:    {split_info['n_days']} 天 ({fmt_date_int(sorted_dates[0])} ~ {fmt_date_int(sorted_dates[-1])})")
    print(f"  训练集:    {split_info['n_train']} 天  {fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}")
    print(f"  验证集:    {split_info['n_valid']} 天  {fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}")
    print(f"  测试集:    {split_info['n_test']} 天  {fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}")
    print(f"  天数比例:  {split_info['n_train']}:{split_info['n_valid']}:{split_info['n_test']}\n")

    # ================================================================
    #  Step 5/9: 过滤低频用户 + 构建全局 ID 映射 + vocab_size
    # ================================================================
    print("[Step 5/9] 过滤低频用户 + 构建 ID 映射")
    valid_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    n_dropped = len(user_counts) - len(valid_users)
    print(f"  有效用户: {len(valid_users):,}  (过滤 {n_dropped:,})")

    sorted_users = sorted(valid_users)
    user_idx_map = {u: i for i, u in enumerate(sorted_users)}

    sorted_items = sorted(item_ids)
    item_idx_map = {it: i + 1 for i, it in enumerate(sorted_items)}
    print(f"  视频数:   {len(item_idx_map):,}")

    vocab_size = {
        "user_index": len(user_idx_map),
        "item_index": len(item_idx_map) + 1,
        "user_id": len(user_idx_map) + 1,
        "item_id": len(item_idx_map) + 1,
        "action": len(name2code) + 1,
        "timestamp": 0,
    }
    for col in USER_STATIC_FEATURES + CONTEXT_FEATURES:
        vocab_size[col] = len(vocabs[col]) + 1

    del user_counts, item_ids, sorted_users, sorted_items
    gc.collect()
    print()

    # ================================================================
    #  Step 6/9: Phase 2 — 逐分区编码 + 按日期切分 + 写 block data/user_info
    # ================================================================
    print("[Step 6/9] 逐分区编码 + 按日期切分 + 写 block data/user_info")

    managers = {
        "train": SplitBlockManager("train", output_dir, train_blocks),
        "valid": SplitBlockManager("valid", output_dir, valid_blocks),
        "test": SplitBlockManager("test", output_dir, test_blocks),
    }

    sample_counts = {"train": 0, "valid": 0, "test": 0}
    max_seq = 0

    # 重新加载 uf_enc 用于 Phase 2 (之前被 del 了)
    # 但 uf_enc 的编码是基于初始 vocab 的 int16，Phase 2 时需要应用 OOV 过滤后的 vocab
    # 所以 uf_enc 仍然需要传递
    uf_phase2 = load_user_features(data_dir)
    uf_enc_phase2 = encode_user_features_to_int(
        uf_phase2, build_all_vocabs(uf_phase2)
    )
    del uf_phase2

    for pid in range(n_user_parts):
        result = process_partition(
            tmp_dir,
            pid,
            valid_users,
            user_idx_map,
            item_idx_map,
            uf_enc_phase2,
            vocabs,
            pat2name,
            name2code,
            valid_start_date,
            test_start_date,
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

    for mgr in managers.values():
        mgr.close_writers()

    total = sum(sample_counts.values())
    print(
        f"\n  train={sample_counts['train']:,}  "
        f"valid={sample_counts['valid']:,}  "
        f"test={sample_counts['test']:,}"
    )
    print(f"  总计={total:,}\n")

    if total == 0:
        raise ValueError(
            "处理后数据为空，请检查原始数据或调小 min_user_interactions。"
        )

    del valid_users, user_idx_map, uf_enc_phase2
    gc.collect()

    # ================================================================
    #  Step 7/9: 为每个 block 构建 item_info
    # ================================================================
    print("\n[Step 7/9] 为每个 block 构建 item_info")

    # 加载原始视频特征表 (用于后续 item vocab 和 global_item_lookup)
    item_feat_raw = load_video_basic_features(data_dir)

    # 构建 item 特征 vocab
    # 为简化实现, 这里对 item 特征不做 OOV 过滤 (仅对 user 特征做了 OOV 过滤)
    # 因为 item 特征的频次统计需要扫描日志中每个 item 出现次数,
    # 而 item 特征值是从独立文件加载的, 与日志频次独立
    item_vs = {}
    for col in ITEM_STATIC_FEATURES:
        item_feat_raw[col] = item_feat_raw[col].astype(str).replace(
            {"nan": "__MISSING__", "": "__MISSING__"}
        )
        vocab = build_ordered_vocab(item_feat_raw[col], start=1)
        item_feat_raw[col] = item_feat_raw[col].map(vocab).astype(np.int32)
        item_vs[col] = len(vocab) + 1

    global_item_lookup = build_global_item_lookup(item_feat_raw, item_idx_map)

    for split_name in ["train", "valid", "test"]:
        print(f"  [Build item_info] {split_name}")
        managers[split_name].write_item_info_blocks(global_item_lookup)
        gc.collect()

    del item_feat_raw, global_item_lookup
    gc.collect()

    # ================================================================
    #  Step 8/9: 保存 meta_data + block_manifest
    # ================================================================
    print("\n[Step 8/9] 保存 meta_data + block_manifest")

    full_vs = dict(vocab_size)
    full_vs.update(item_vs)

    meta = {
        "sample_size": {
            "total": int(total),
            "train": int(sample_counts["train"]),
            "valid": int(sample_counts["valid"]),
            "test": int(sample_counts["test"]),
        },
        "split_by_date": {
            "train_days": split_info["n_train"],
            "train_range": f"{fmt_date_int(train_dates[0])} ~ {fmt_date_int(train_dates[-1])}",
            "valid_days": split_info["n_valid"],
            "valid_range": f"{fmt_date_int(valid_dates[0])} ~ {fmt_date_int(valid_dates[-1])}",
            "test_days": split_info["n_test"],
            "test_range": f"{fmt_date_int(test_dates[0])} ~ {fmt_date_int(test_dates[-1])}",
        },
        "oov_filter": {
            "min_feat_count": int(min_feat_count),
            "applied_to": "USER_STATIC_FEATURES (基于日志中用户出现次数统计)",
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
        "vocab_size": {k: int(v) for k, v in full_vs.items()},
        "label": LABEL_COLUMNS,
        "action_vocab": {k: int(v) for k, v in name2code.items()},
        "action_vocab_desc": (
            "编码后的 action 词表，用于 dataloader 基于 full_action_seq "
            "构造 task-specific token masks。"
        ),
        "user_info_schema": {
            "fields": [
                "user_index",
                "full_item_seq",
                "full_action_seq",
                "full_timestamp_seq",
            ],
            "full_timestamp_seq_desc": "按时间顺序排列的原始 time_ms 序列",
            "desc": (
                "这里的 user_index / full_item_seq 中的 item index 都是 block-local index；"
                "full_action_seq / full_timestamp_seq 为全局时间顺序序列。"
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
            "full_timestamp_seq": int(max_seq),
        },
    }

    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    print("  [Saved] meta_data.json")

    block_manifest = {
        "train": managers["train"].buildmanifest(),
        "valid": managers["valid"].buildmanifest(),
        "test": managers["test"].buildmanifest(),
    }
    with open(output_dir / "block_manifest.json", "w", encoding="utf-8") as f:
        json.dump(block_manifest, f, ensure_ascii=False, indent=4)
    print("  [Saved] block_manifest.json")

    # ================================================================
    #  Step 9/9: 清理临时文件
    # ================================================================
    print("\n[Step 9/9] 清理临时分区文件")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  Done.")

    # ================================================================
    #  Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("  KuaiRand-27K Preprocess Done (按日期比例切分 + blocked 输出 + OOV 过滤)")
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
    print(json.dumps(meta, ensure_ascii=False, indent=4))


# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess KuaiRand-27K seq-action data into blocked train/valid/test with OOV filtering."
    )
    parser.add_argument("--data_dir", type=str, default="/mnt/ceph-nj1-csp/bingoozhang/salmonli/data/KuaiRand-27K/data")
    parser.add_argument("--output_dir", type=str, default="/mnt/ceph-nj1-csp/bingoozhang/salmonli/data/KuaiRand_Video_Action")
    parser.add_argument("--min_user_interactions", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--n_user_parts", type=int, default=8,
                        help="Phase1 按 user hash 的临时分区数")
    parser.add_argument("--chunk_size", type=int, default=4_000_000,
                        help="读取 CSV 时单次处理多少行")
    parser.add_argument("--buffer_flush_size", type=int, default=1_000_000,
                        help="临时分区缓存多少 interaction 后落盘")
    parser.add_argument("--train_blocks", type=int, default=16)
    parser.add_argument("--valid_blocks", type=int, default=2)
    parser.add_argument("--test_blocks", type=int, default=2)
    parser.add_argument("--min_feat_count", type=int, default=MIN_FEAT_COUNT,
                        help="OOV 过滤阈值: 特征值出现次数 < 此值的统一映射为 0")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="若输出目录已存在，则删除后重建")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_and_split(
        data_dir=args.data_dir,
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