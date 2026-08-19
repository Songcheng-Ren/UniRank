#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the pure-CTR Taobao pipeline from ad impressions only."""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.Taobao.preprocess_Taobao_seq_action import (
    MIN_FEAT_COUNT,
    preprocess_and_split,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Taobao impressions for pure CTR prediction."
    )
    parser.add_argument("--data_dir", type=str, default="./Taobao")
    parser.add_argument("--output_dir", type=str, default="./Taobao_CTR")
    parser.add_argument("--min_user_interactions", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--n_user_parts", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=4_000_000)
    parser.add_argument("--buffer_flush_size", type=int, default=1_000_000)
    parser.add_argument("--train_blocks", type=int, default=32)
    parser.add_argument("--valid_blocks", type=int, default=8)
    parser.add_argument("--test_blocks", type=int, default=8)
    parser.add_argument("--min_feat_count", type=int, default=MIN_FEAT_COUNT)
    parser.add_argument(
        "--timestamp_unit", choices=["auto", "s", "ms"], default="auto"
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
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
        timestamp_unit=args.timestamp_unit,
        use_behavior_labels=False,
        task_mode="ctr",
        overwrite=args.overwrite,
    )
