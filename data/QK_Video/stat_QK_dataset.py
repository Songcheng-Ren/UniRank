#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 QK-Video blocked 数据集信息。"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset_stats_utils import DatasetSpec, run_cli


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FIELDS = [
    "user_index", "item_index", "seq_len",
    "user_id", "item_id", "video_category",
    "watching_times", "gender", "age", "action",
]

LABEL_COLUMNS = ["click", "follow", "like", "share"]


if __name__ == "__main__":
    run_cli(
        DatasetSpec(
            name="QK_Video_Action",
            default_output_dir=ROOT / "QK_Video_Action",
            feature_fields=FEATURE_FIELDS,
            label_columns=LABEL_COLUMNS,
        )
    )
