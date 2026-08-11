#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert KuaiRec into UniRank's blocked sequential-action format.

Two protocols are supported:

* ``big_chrono`` uses only ``big_matrix.csv`` and splits its dates into
  chronological train/validation/test ranges.
* ``official_dense`` uses the early/late dates of ``big_matrix.csv`` for
  train/validation and reserves every row of the nearly fully-observed
  ``small_matrix.csv`` for test.

KuaiRec does not expose click or like feedback. The first UniRank version uses
the binary target recommended by the dataset authors: ``watch_ratio > 2.0``.
Historical actions encode either a non-positive exposure or ``high_watch``.

Only static columns are taken from ``item_daily_features.csv``. Aggregated
show/play/like counters are deliberately excluded because putting future daily
statistics into static item_info would leak target-period information.
"""

import argparse
import ast
import gc
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.KuaiRand.preprocess_Kuairand_seq_action import (  # noqa: E402
    SplitBlockManager as BaseSplitBlockManager,
    build_date_split,
)


PROTOCOLS = ("big_chrono", "official_dense")
LABEL_COLUMNS = ["high_watch"]
ACTION_VOCAB = {"exposure": 1, "high_watch": 2}

USER_STATIC_FEATURES = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]

ITEM_STATIC_FEATURES = [
    "author_id", "video_type", "upload_type", "duration_bucket",
    "category_1", "category_2",
]
CONTEXT_FEATURES = ["day_of_week", "is_weekend", "hour"]

FINAL_COLUMNS = (
    ["user_index", "item_index", "seq_len", "user_id"]
    + USER_STATIC_FEATURES
    + CONTEXT_FEATURES
    + LABEL_COLUMNS
)

DURATION_BINS = [-np.inf, 0, 3000, 7000, 15000, 30000, 60000, np.inf]
DURATION_LABELS = [
    "missing", "0-3s", "3-7s", "7-15s", "15-30s", "30-60s", "60s+",
]


class SplitBlockManager(BaseSplitBlockManager):
    """KuaiRec block writer with its own static item schema."""

    def write_item_info_blocks(self, global_item_lookup):
        for block_id in range(self.num_blocks):
            if self.block_rows[block_id] == 0:
                continue

            item_map = self.item_maps[block_id]
            local_size = max(item_map.values()) if item_map else 0
            global_indices = np.zeros(local_size + 1, dtype=np.int32)
            for global_index, local_index in item_map.items():
                global_indices[local_index] = np.int32(global_index)

            output = {
                "item_index": np.arange(local_size + 1, dtype=np.int32),
                "item_id": global_indices.copy(),
            }
            if local_size > 0:
                features = global_item_lookup.reindex(
                    global_indices[1:]
                ).fillna(0)
                output["item_id"][1:] = features["item_id"].to_numpy(
                    dtype=np.int32, copy=False
                )
                for column in ITEM_STATIC_FEATURES:
                    values = np.zeros(local_size + 1, dtype=np.int32)
                    values[1:] = features[column].to_numpy(
                        dtype=np.int32, copy=False
                    )
                    output[column] = values
            else:
                for column in ITEM_STATIC_FEATURES:
                    output[column] = np.zeros(1, dtype=np.int32)

            path = self.item_info_dir / f"part-{block_id:05d}.parquet"
            pd.DataFrame(output).to_parquet(
                path, index=False, engine="pyarrow"
            )


def resolve_raw_data_dir(data_dir):
    """Accept the archive root, ``KuaiRec`` root, or its ``data`` directory."""
    root = Path(data_dir).expanduser().resolve()
    candidates = [root, root / "data", root / "KuaiRec" / "data"]
    for candidate in candidates:
        if (candidate / "big_matrix.csv").exists():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"big_matrix.csv was not found. Tried: {tried}")


def _clean_categorical(series):
    missing = series.isna()
    values = series.astype(str).str.strip()
    values = values.mask(missing | values.isin(["", "nan", "None"]), "__MISSING__")
    return values


def _encode_categorical_frame(frame, columns):
    encoded = frame.copy()
    vocab_sizes = {}
    for column in columns:
        if column not in encoded:
            encoded[column] = "__MISSING__"
        values = _clean_categorical(encoded[column])
        categories = sorted(value for value in values.unique() if value != "__MISSING__")
        vocab = {value: index + 1 for index, value in enumerate(categories)}
        encoded[column] = values.map(vocab).fillna(0).astype(np.int32)
        vocab_sizes[column] = len(vocab) + 1
    return encoded, vocab_sizes


def _local_datetime(timestamp_seconds):
    utc = pd.to_datetime(timestamp_seconds, unit="s", utc=True, errors="coerce")
    return utc.dt.tz_convert("Asia/Shanghai")


def _normalize_matrix_chunk(raw, source, order_offset):
    required = {"user_id", "video_id", "timestamp", "watch_ratio"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"KuaiRec matrix is missing required columns: {missing}")

    frame = pd.DataFrame()
    frame["raw_user_id"] = pd.to_numeric(raw["user_id"], errors="coerce")
    frame["raw_video_id"] = pd.to_numeric(raw["video_id"], errors="coerce")
    frame["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    frame["watch_ratio"] = pd.to_numeric(raw["watch_ratio"], errors="coerce")
    if "video_duration" in raw:
        frame["video_duration"] = pd.to_numeric(
            raw["video_duration"], errors="coerce"
        ).fillna(0)
    else:
        frame["video_duration"] = 0.0

    frame = frame.dropna(
        subset=["raw_user_id", "raw_video_id", "timestamp", "watch_ratio"]
    ).copy()
    frame["raw_user_id"] = frame["raw_user_id"].astype(np.int64)
    frame["raw_video_id"] = frame["raw_video_id"].astype(np.int64)
    frame["timestamp_ms"] = np.rint(frame["timestamp"] * 1000).astype(np.int64)
    frame["watch_ratio"] = frame["watch_ratio"].astype(np.float32)
    frame["video_duration"] = frame["video_duration"].astype(np.float32)
    frame["source"] = np.int8(source)
    frame["event_order"] = np.arange(
        order_offset, order_offset + len(frame), dtype=np.int64
    )

    if "date" in raw:
        aligned_date = pd.to_numeric(raw.loc[frame.index, "date"], errors="coerce")
    else:
        aligned_date = pd.Series(np.nan, index=frame.index)
    derived_date = _local_datetime(frame["timestamp"]).dt.strftime("%Y%m%d")
    frame["date"] = aligned_date.fillna(derived_date).astype(np.int32).to_numpy()
    frame.drop(columns=["timestamp"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def partition_matrices(raw_dir, temp_dir, protocol, n_user_parts, chunk_size):
    """Write user-hash partitions while collecting compact global statistics."""
    matrix_specs = [(raw_dir / "big_matrix.csv", 0, "big")]
    if protocol == "official_dense":
        small_path = raw_dir / "small_matrix.csv"
        if not small_path.exists():
            raise FileNotFoundError(
                "official_dense requires small_matrix.csv, but it was not found"
            )
        matrix_specs.append((small_path, 1, "small"))

    partition_root = temp_dir / "matrix_parts"
    partition_root.mkdir(parents=True, exist_ok=True)
    user_counts = Counter()
    item_ids = set()
    source_dates = {"big": set(), "small": set()}
    duration_hints = {}
    source_rows = {"big": 0, "small": 0}

    for matrix_path, source, source_name in matrix_specs:
        header = pd.read_csv(matrix_path, nrows=0).columns.tolist()
        desired = [
            "user_id", "video_id", "timestamp", "date", "watch_ratio",
            "video_duration",
        ]
        usecols = [column for column in desired if column in header]
        order_offset = 0
        for chunk_index, raw in enumerate(
            pd.read_csv(matrix_path, usecols=usecols, chunksize=chunk_size)
        ):
            frame = _normalize_matrix_chunk(raw, source, order_offset)
            order_offset += len(frame)
            source_rows[source_name] += len(frame)

            user_counts.update(
                {int(k): int(v) for k, v in frame["raw_user_id"].value_counts().items()}
            )
            item_ids.update(int(value) for value in frame["raw_video_id"].unique())
            source_dates[source_name].update(int(value) for value in frame["date"].unique())

            duration_rows = frame.loc[
                frame["video_duration"] > 0,
                ["raw_video_id", "video_duration"],
            ].drop_duplicates("raw_video_id")
            for row in duration_rows.itertuples(index=False):
                duration_hints.setdefault(int(row.raw_video_id), float(row.video_duration))

            partition_ids = np.mod(frame["raw_user_id"].to_numpy(), n_user_parts)
            frame["partition_id"] = partition_ids.astype(np.int16)
            for partition_id, part in frame.groupby("partition_id", sort=False):
                part_dir = partition_root / f"part-{int(partition_id):05d}"
                part_dir.mkdir(parents=True, exist_ok=True)
                output = part_dir / f"{source_name}-{chunk_index:05d}.parquet"
                part.drop(columns=["partition_id"]).to_parquet(
                    output, index=False, engine="pyarrow"
                )
            del raw, frame
            gc.collect()
            print(
                f"  {source_name}: chunk={chunk_index + 1}, "
                f"rows={source_rows[source_name]:,}"
            )

    return {
        "partition_root": partition_root,
        "user_counts": user_counts,
        "item_ids": item_ids,
        "source_dates": source_dates,
        "duration_hints": duration_hints,
        "source_rows": source_rows,
    }


def load_user_lookup(raw_dir, user_id_map):
    path = raw_dir / "user_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required KuaiRec user feature file is missing: {path}")
    user_features = pd.read_csv(path)
    if "user_id" not in user_features:
        raise ValueError("user_features.csv is missing user_id")
    user_features["raw_user_id"] = pd.to_numeric(
        user_features["user_id"], errors="coerce"
    )
    user_features = user_features.dropna(subset=["raw_user_id"]).copy()
    user_features["raw_user_id"] = user_features["raw_user_id"].astype(np.int64)
    user_features = user_features[user_features["raw_user_id"].isin(user_id_map)]
    selected = user_features[["raw_user_id"]].copy()
    for column in USER_STATIC_FEATURES:
        selected[column] = (
            user_features[column] if column in user_features else "__MISSING__"
        )
    selected, vocab_sizes = _encode_categorical_frame(
        selected, USER_STATIC_FEATURES
    )
    selected = selected.drop_duplicates("raw_user_id", keep="last")
    return selected, vocab_sizes


def _parse_category_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    elif pd.isna(value):
        parsed = []
    else:
        try:
            parsed = ast.literal_eval(str(value))
            if not isinstance(parsed, (list, tuple)):
                parsed = [parsed]
        except (ValueError, SyntaxError):
            parsed = []
    parsed = [item for item in parsed if not pd.isna(item)]
    first = parsed[0] if len(parsed) > 0 else "__MISSING__"
    second = parsed[1] if len(parsed) > 1 else "__MISSING__"
    return first, second


def load_item_lookup(raw_dir, item_id_map, duration_hints):
    item_ids = sorted(item_id_map)
    item_features = pd.DataFrame({"raw_video_id": item_ids})

    daily_path = raw_dir / "item_daily_features.csv"
    if daily_path.exists():
        header = pd.read_csv(daily_path, nrows=0).columns.tolist()
        desired = [
            "video_id", "date", "author_id", "video_type", "upload_type",
            "video_duration",
        ]
        usecols = [column for column in desired if column in header]
        daily = pd.read_csv(daily_path, usecols=usecols)
        if "video_id" not in daily:
            raise ValueError("item_daily_features.csv is missing video_id")
        daily["raw_video_id"] = pd.to_numeric(daily["video_id"], errors="coerce")
        daily = daily.dropna(subset=["raw_video_id"]).copy()
        daily["raw_video_id"] = daily["raw_video_id"].astype(np.int64)
        daily = daily[daily["raw_video_id"].isin(item_id_map)]
        if "date" in daily:
            daily = daily.sort_values(["raw_video_id", "date"])
        daily = daily.groupby("raw_video_id", as_index=False).first()
        keep = [
            column for column in [
                "raw_video_id", "author_id", "video_type", "upload_type",
                "video_duration",
            ] if column in daily
        ]
        item_features = item_features.merge(daily[keep], on="raw_video_id", how="left")

    categories_path = raw_dir / "item_categories.csv"
    if categories_path.exists():
        categories = pd.read_csv(categories_path)
        if {"video_id", "feat"}.issubset(categories.columns):
            categories["raw_video_id"] = pd.to_numeric(
                categories["video_id"], errors="coerce"
            )
            categories = categories.dropna(subset=["raw_video_id"]).copy()
            categories["raw_video_id"] = categories["raw_video_id"].astype(np.int64)
            parsed = categories["feat"].map(_parse_category_list)
            categories["category_1"] = parsed.map(lambda pair: pair[0])
            categories["category_2"] = parsed.map(lambda pair: pair[1])
            item_features = item_features.merge(
                categories[["raw_video_id", "category_1", "category_2"]]
                .drop_duplicates("raw_video_id", keep="last"),
                on="raw_video_id",
                how="left",
            )

    for column in ["author_id", "video_type", "upload_type", "category_1", "category_2"]:
        if column not in item_features:
            item_features[column] = "__MISSING__"

    hint_series = item_features["raw_video_id"].map(duration_hints).fillna(0)
    if "video_duration" in item_features:
        duration = pd.to_numeric(item_features["video_duration"], errors="coerce")
        duration = duration.where(duration > 0, hint_series)
    else:
        duration = hint_series
    item_features["duration_bucket"] = pd.cut(
        duration.fillna(0),
        bins=DURATION_BINS,
        labels=DURATION_LABELS,
        include_lowest=True,
    ).astype(str)

    item_features, vocab_sizes = _encode_categorical_frame(
        item_features, ITEM_STATIC_FEATURES
    )
    item_features["item_index"] = item_features["raw_video_id"].map(item_id_map)
    item_features["item_id"] = item_features["item_index"]
    item_features = item_features[
        ["item_index", "item_id"] + ITEM_STATIC_FEATURES
    ].drop_duplicates("item_index", keep="last")
    item_features = item_features.set_index("item_index").sort_index()
    return item_features, vocab_sizes


def build_split_info(protocol, source_dates, train_ratio, valid_ratio, test_ratio):
    big_dates = sorted(source_dates["big"])
    if protocol == "big_chrono":
        info = build_date_split(big_dates, train_ratio, valid_ratio, test_ratio)
        return {
            "train_dates": info["train_dates"],
            "valid_dates": info["valid_dates"],
            "test_dates": info["test_dates"],
            "valid_start_date": int(info["valid_start_date"]),
            "test_start_date": int(info["test_start_date"]),
        }

    if len(big_dates) < 2:
        raise ValueError("official_dense requires at least two dates in big_matrix")
    denominator = train_ratio + valid_ratio
    train_share = train_ratio / denominator if denominator > 0 else 0.9
    train_days = min(max(int(round(len(big_dates) * train_share)), 1), len(big_dates) - 1)
    return {
        "train_dates": big_dates[:train_days],
        "valid_dates": big_dates[train_days:],
        "test_dates": sorted(source_dates["small"]),
        "valid_start_date": int(big_dates[train_days]),
        "test_start_date": None,
    }


def _add_model_columns(frame, user_id_map, item_id_map, user_lookup, threshold):
    frame = frame.copy()
    frame["user_index"] = frame["raw_user_id"].map(user_id_map)
    frame["item_index"] = frame["raw_video_id"].map(item_id_map)
    frame = frame.dropna(subset=["user_index", "item_index"]).copy()
    frame["user_index"] = frame["user_index"].astype(np.int32)
    frame["item_index"] = frame["item_index"].astype(np.int32)
    frame["user_id"] = (frame["user_index"] + 1).astype(np.int32)
    frame["high_watch"] = (frame["watch_ratio"] > threshold).astype(np.float32)
    frame["action"] = np.where(frame["high_watch"] > 0, 2, 1).astype(np.int32)

    local_time = _local_datetime(frame["timestamp_ms"] / 1000.0)
    frame["day_of_week"] = (local_time.dt.dayofweek + 1).astype(np.int32)
    frame["is_weekend"] = (local_time.dt.dayofweek >= 5).astype(np.int32) + 1
    frame["hour"] = (local_time.dt.hour + 1).astype(np.int32)

    frame = frame.merge(user_lookup, on="raw_user_id", how="left")
    for column in USER_STATIC_FEATURES:
        frame[column] = frame[column].fillna(0).astype(np.int32)
    return frame.sort_values(
        ["user_index", "timestamp_ms", "source", "event_order"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_user_info(frame):
    rows = []
    for user_index, group in frame.groupby("user_index", sort=True):
        rows.append({
            "user_index": int(user_index),
            "full_item_seq": group["item_index"].astype(int).tolist(),
            "full_action_seq": group["action"].astype(int).tolist(),
            "full_timestamp_seq": group["timestamp_ms"].astype(np.int64).tolist(),
        })
    return rows


def _select_final(frame):
    if len(frame) == 0:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    return frame[FINAL_COLUMNS].reset_index(drop=True)


def process_partition(
    files,
    protocol,
    valid_users,
    user_id_map,
    item_id_map,
    user_lookup,
    threshold,
    valid_start_date,
    test_start_date,
):
    frames = [pd.read_parquet(path) for path in files]
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["raw_user_id"].isin(valid_users)]
    if len(frame) == 0:
        return None
    frame = _add_model_columns(
        frame, user_id_map, item_id_map, user_lookup, threshold
    )

    if protocol == "big_chrono":
        frame["seq_len"] = frame.groupby("user_index", sort=False).cumcount().astype(np.int32)
        user_info = _build_user_info(frame)
        return {
            "train": _select_final(frame[frame["date"] < valid_start_date]),
            "valid": _select_final(
                frame[(frame["date"] >= valid_start_date) & (frame["date"] < test_start_date)]
            ),
            "test": _select_final(frame[frame["date"] >= test_start_date]),
            "user_info": {
                "train": user_info,
                "valid": user_info,
                "test": user_info,
            },
        }

    big = frame[frame["source"] == 0].copy()
    big["seq_len"] = big.groupby("user_index", sort=False).cumcount().astype(np.int32)
    big_user_info = _build_user_info(big)

    combined = frame.copy()
    combined["seq_len"] = combined.groupby("user_index", sort=False).cumcount().astype(np.int32)
    combined_user_info = _build_user_info(combined)
    test = combined[combined["source"] == 1]
    return {
        "train": _select_final(big[big["date"] < valid_start_date]),
        "valid": _select_final(big[big["date"] >= valid_start_date]),
        "test": _select_final(test),
        "user_info": {
            "train": big_user_info,
            "valid": big_user_info,
            "test": combined_user_info,
        },
    }


def _feature_specs(vocab_sizes):
    specs = [
        {"name": "user_index", "active": True, "dtype": "int", "type": "meta"},
        {"name": "item_index", "active": True, "dtype": "int", "type": "meta"},
        {"name": "seq_len", "active": True, "dtype": "int", "type": "meta"},
        {
            "name": "user_id", "active": True, "dtype": "int",
            "type": "categorical", "vocab_size": int(vocab_sizes["user_id"]),
        },
    ]
    for column in ["item_id"] + ITEM_STATIC_FEATURES:
        specs.append({
            "name": column,
            "active": True,
            "dtype": "int",
            "type": "categorical",
            "vocab_size": int(vocab_sizes[column]),
            "source": "item",
        })
    for column in USER_STATIC_FEATURES:
        specs.append({
            "name": column,
            "active": True,
            "dtype": "int",
            "type": "categorical",
            "vocab_size": int(vocab_sizes[column]),
        })
    fixed_context_sizes = {"day_of_week": 8, "is_weekend": 3, "hour": 25}
    for column in CONTEXT_FEATURES:
        specs.append({
            "name": column,
            "active": True,
            "dtype": "int",
            "type": "categorical",
            "vocab_size": fixed_context_sizes[column],
        })
    specs.append({
        "name": "action", "active": True, "dtype": "int",
        "type": "categorical", "vocab_size": 3, "source": "action",
    })
    return specs


def write_dataset_config_snippet(output_dir, dataset_id, vocab_sizes):
    dataset = {
        "data_root": str(output_dir.parent),
        "data_format": "parquet",
        "train_data": str(output_dir / "train" / "data"),
        "valid_data": str(output_dir / "valid" / "data"),
        "test_data": str(output_dir / "test" / "data"),
        "train_user_info": str(output_dir / "train" / "user_info"),
        "train_item_info": str(output_dir / "train" / "item_info"),
        "valid_user_info": str(output_dir / "valid" / "user_info"),
        "valid_item_info": str(output_dir / "valid" / "item_info"),
        "test_user_info": str(output_dir / "test" / "user_info"),
        "test_item_info": str(output_dir / "test" / "item_info"),
        "rebuild_dataset": False,
        "blocked": True,
        "block_cache_size": 2,
        "feature_cols": _feature_specs(vocab_sizes),
        "label_col": [{"name": "high_watch", "dtype": "float"}],
    }
    path = output_dir / "dataset_config_snippet.yaml"
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            {dataset_id: dataset}, file, sort_keys=False, allow_unicode=True
        )
    return path


def preprocess_and_split(
    data_dir,
    output_dir,
    protocol="big_chrono",
    dataset_id=None,
    positive_threshold=2.0,
    min_user_interactions=5,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    n_user_parts=32,
    chunk_size=1_000_000,
    train_blocks=32,
    valid_blocks=8,
    test_blocks=8,
    overwrite=False,
):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol={protocol!r}; expected one of {PROTOCOLS}")
    if positive_threshold < 0:
        raise ValueError("positive_threshold must be non-negative")
    if min(n_user_parts, train_blocks, valid_blocks, test_blocks, chunk_size) <= 0:
        raise ValueError("partition, block, and chunk sizes must be positive")

    raw_dir = resolve_raw_data_dir(data_dir)
    output_dir = Path(output_dir).expanduser().resolve()
    dataset_id = dataset_id or (
        "KuaiRec_Big_Watch_Action"
        if protocol == "big_chrono"
        else "KuaiRec_Dense_Watch_Action"
    )
    if output_dir.name != dataset_id:
        raise ValueError(
            "UniRank expects output_dir.name to equal dataset_id so FeatureProcessor "
            f"can locate feature_map.json; got output_dir={output_dir}, "
            f"dataset_id={dataset_id!r}"
        )

    if output_dir == raw_dir or raw_dir.is_relative_to(output_dir):
        raise ValueError("output_dir cannot equal or contain the raw KuaiRec directory")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Partition interaction matrices ({protocol})")
    phase1 = partition_matrices(
        raw_dir, temp_dir, protocol, n_user_parts, chunk_size
    )

    valid_users = {
        int(user_id) for user_id, count in phase1["user_counts"].items()
        if count >= min_user_interactions
    }
    if not valid_users:
        raise ValueError("No users remain after min_user_interactions filtering")
    sorted_users = sorted(valid_users)
    sorted_items = sorted(phase1["item_ids"])
    user_id_map = {user_id: index for index, user_id in enumerate(sorted_users)}
    item_id_map = {item_id: index + 1 for index, item_id in enumerate(sorted_items)}

    print("[2/7] Encode static user and item features")
    user_lookup, user_vocab_sizes = load_user_lookup(raw_dir, user_id_map)
    item_lookup, item_vocab_sizes = load_item_lookup(
        raw_dir, item_id_map, phase1["duration_hints"]
    )
    vocab_sizes = {
        "user_id": len(user_id_map) + 1,
        "item_id": len(item_id_map) + 1,
        "action": 3,
        **user_vocab_sizes,
        **item_vocab_sizes,
    }

    print("[3/7] Build chronological split boundaries")
    split_info = build_split_info(
        protocol,
        phase1["source_dates"],
        train_ratio,
        valid_ratio,
        test_ratio,
    )

    managers = {
        "train": SplitBlockManager("train", output_dir, train_blocks),
        "valid": SplitBlockManager("valid", output_dir, valid_blocks),
        "test": SplitBlockManager("test", output_dir, test_blocks),
    }
    sample_counts = {"train": 0, "valid": 0, "test": 0}
    max_sequence_length = 0

    print("[4/7] Encode user partitions and write blocked data")
    partition_root = phase1["partition_root"]
    for partition_id in range(n_user_parts):
        files = sorted(
            (partition_root / f"part-{partition_id:05d}").glob("*.parquet")
        )
        if not files:
            continue
        result = process_partition(
            files=files,
            protocol=protocol,
            valid_users=valid_users,
            user_id_map=user_id_map,
            item_id_map=item_id_map,
            user_lookup=user_lookup,
            threshold=positive_threshold,
            valid_start_date=split_info["valid_start_date"],
            test_start_date=split_info["test_start_date"],
        )
        if result is None:
            continue
        for split_name in ("train", "valid", "test"):
            data = result[split_name]
            if len(data) == 0:
                continue
            user_info = result["user_info"][split_name]
            managers[split_name].append_partition(
                pid=partition_id,
                split_df=data,
                user_info_rows=user_info,
            )
            sample_counts[split_name] += len(data)
            for row in user_info:
                max_sequence_length = max(
                    max_sequence_length, len(row["full_item_seq"])
                )
        print(
            f"  partition={partition_id + 1}/{n_user_parts}, "
            f"samples={sum(sample_counts.values()):,}"
        )
        del result
        gc.collect()

    for manager in managers.values():
        manager.close_writers()
    for split_name, count in sample_counts.items():
        if count == 0:
            raise ValueError(f"The generated {split_name} split is empty")

    print("[5/7] Write block-local item_info tables")
    for manager in managers.values():
        manager.write_item_info_blocks(item_lookup)

    print("[6/7] Write metadata and exact dataset config snippet")
    meta = {
        "dataset_id": dataset_id,
        "source": "KuaiRec",
        "protocol": protocol,
        "positive_label": {
            "name": "high_watch",
            "definition": f"watch_ratio > {positive_threshold}",
            "threshold": float(positive_threshold),
        },
        "sample_size": {
            "total": int(sum(sample_counts.values())),
            **{name: int(value) for name, value in sample_counts.items()},
        },
        "raw_source_rows": phase1["source_rows"],
        "split": {
            "train_dates": [int(value) for value in split_info["train_dates"]],
            "valid_dates": [int(value) for value in split_info["valid_dates"]],
            "test_dates": [int(value) for value in split_info["test_dates"]],
            "official_dense_note": (
                "small_matrix is test-only; train/valid histories contain big_matrix only"
                if protocol == "official_dense" else None
            ),
        },
        "blocked_layout": {
            "train_blocks": int(train_blocks),
            "valid_blocks": int(valid_blocks),
            "test_blocks": int(test_blocks),
            "local_index_rule": {
                "user_index": "block-local dense index starting from 0",
                "item_index": "block-local dense index with 0 reserved for padding",
                "user_id": "global embedding id with 0 reserved",
                "item_id": "global embedding id with 0 reserved",
            },
        },
        "vocab_size": {key: int(value) for key, value in vocab_sizes.items()},
        "label": LABEL_COLUMNS,
        "action_vocab": ACTION_VOCAB,
        "user_info_schema": {
            "fields": [
                "user_index", "full_item_seq", "full_action_seq",
                "full_timestamp_seq",
            ]
        },
        "item_info_schema": {
            "fields": ["item_index", "item_id"] + ITEM_STATIC_FEATURES,
            "daily_feature_policy": (
                "Only earliest available static descriptors are used; "
                "engagement counters are excluded to prevent leakage"
            ),
        },
        "max_len": {
            "full_item_seq": int(max_sequence_length),
            "full_action_seq": int(max_sequence_length),
            "full_timestamp_seq": int(max_sequence_length),
        },
    }
    with open(output_dir / "meta_data.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    manifest = {
        split_name: managers[split_name].buildmanifest()
        for split_name in ("train", "valid", "test")
    }
    with open(output_dir / "block_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    snippet_path = write_dataset_config_snippet(
        output_dir, dataset_id, vocab_sizes
    )

    print("[7/7] Clean temporary partitions")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Done: {output_dir}")
    print(f"Dataset config snippet: {snippet_path}")
    print(json.dumps(meta["sample_size"], ensure_ascii=False, indent=2))
    return meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess KuaiRec into UniRank blocked sequential data"
    )
    parser.add_argument("--data_dir", default="./data/KuaiRec/raw")
    parser.add_argument("--output_dir", default="./data/KuaiRec_Big_Watch_Action")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="big_chrono")
    parser.add_argument("--dataset_id", default=None)
    parser.add_argument("--positive_threshold", type=float, default=2.0)
    parser.add_argument("--min_user_interactions", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--n_user_parts", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--train_blocks", type=int, default=32)
    parser.add_argument("--valid_blocks", type=int, default=8)
    parser.add_argument("--test_blocks", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_and_split(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        protocol=args.protocol,
        dataset_id=args.dataset_id,
        positive_threshold=args.positive_threshold,
        min_user_interactions=args.min_user_interactions,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        n_user_parts=args.n_user_parts,
        chunk_size=args.chunk_size,
        train_blocks=args.train_blocks,
        valid_blocks=args.valid_blocks,
        test_blocks=args.test_blocks,
        overwrite=args.overwrite,
    )
