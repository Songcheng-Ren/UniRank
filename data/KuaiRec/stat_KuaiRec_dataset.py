#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Statistics for a preprocessed KuaiRec blocked dataset."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset_stats_utils import DatasetSpec, run_cli


ROOT = Path(__file__).resolve().parents[1]

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

FEATURE_FIELDS = (
    ["user_index", "item_index", "seq_len", "user_id", "item_id"]
    + ITEM_STATIC_FEATURES
    + USER_STATIC_FEATURES
    + CONTEXT_FEATURES
    + ["action"]
)


if __name__ == "__main__":
    run_cli(
        DatasetSpec(
            name="KuaiRec_Watch_Action",
            default_output_dir=ROOT / "KuaiRec_Big_Watch_Action",
            feature_fields=FEATURE_FIELDS,
            label_columns=["high_watch"],
        )
    )
