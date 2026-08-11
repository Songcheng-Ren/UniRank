import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.KuaiRec.preprocess_KuaiRec_seq_action import preprocess_and_split
from model_zoo.QFormerCross3 import QFormerCross3
from unirank.features import FeatureMap
from unirank.preprocess import FeatureProcessor, build_dataset
from unirank.pytorch.dataloaders import RankDataLoader


class KuaiRecPreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="unirank-kuairec-test-")
        cls.root = Path(cls.temp_dir.name)
        cls.raw_dir = cls.root / "raw" / "KuaiRec" / "data"
        cls.raw_dir.mkdir(parents=True)
        cls._write_raw_fixture()

        cls.big_output = cls.root / "KuaiRec_Big_Watch_Action"
        cls.dense_output = cls.root / "KuaiRec_Dense_Watch_Action"
        with contextlib.redirect_stdout(io.StringIO()):
            preprocess_and_split(
                data_dir=cls.raw_dir,
                output_dir=cls.big_output,
                protocol="big_chrono",
                min_user_interactions=2,
                n_user_parts=2,
                chunk_size=11,
                train_blocks=2,
                valid_blocks=1,
                test_blocks=1,
            )
            preprocess_and_split(
                data_dir=cls.raw_dir,
                output_dir=cls.dense_output,
                protocol="official_dense",
                min_user_interactions=2,
                n_user_parts=2,
                chunk_size=11,
                train_blocks=2,
                valid_blocks=1,
                test_blocks=1,
            )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @classmethod
    def _write_raw_fixture(cls):
        start = pd.Timestamp("2020-07-01 08:00:00", tz="Asia/Shanghai")
        big_rows = []
        for day in range(5):
            for user_id in range(4):
                for offset in range(2):
                    timestamp = start + pd.Timedelta(
                        days=day, minutes=user_id * 10 + offset
                    )
                    video_id = (user_id + day + offset) % 5
                    big_rows.append({
                        "user_id": user_id,
                        "video_id": video_id,
                        "play_duration": 10000,
                        "video_duration": 5000 + video_id * 1000,
                        "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "date": int(timestamp.strftime("%Y%m%d")),
                        "timestamp": timestamp.timestamp(),
                        "watch_ratio": (
                            3.0 if (day + user_id + offset) % 2 else 0.8
                        ),
                    })
        pd.DataFrame(big_rows).to_csv(
            cls.raw_dir / "big_matrix.csv", index=False
        )

        small_rows = []
        for user_id in range(2):
            for offset in range(3):
                # Interleave dense-test events with the big timeline. This
                # catches accidental small_matrix leakage into train history.
                timestamp = start + pd.Timedelta(
                    days=1 + offset, minutes=user_id * 10 + 5
                )
                small_rows.append({
                    "user_id": user_id,
                    "video_id": (user_id + offset) % 5,
                    "play_duration": 12000,
                    "video_duration": 6000,
                    "time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "date": int(timestamp.strftime("%Y%m%d")),
                    "timestamp": timestamp.timestamp(),
                    "watch_ratio": 2.5 if offset % 2 else 0.5,
                })
        pd.DataFrame(small_rows).to_csv(
            cls.raw_dir / "small_matrix.csv", index=False
        )

        user_rows = []
        for user_id in range(4):
            row = {
                "user_id": user_id,
                "user_active_degree": "high_active",
                "is_lowactive_period": user_id % 2,
                "is_live_streamer": 0,
                "is_video_author": 1,
                "follow_user_num_range": "(0,10]",
                "fans_user_num_range": "0",
                "friend_user_num_range": "[1,5)",
                "register_days_range": "31-60",
            }
            row.update({
                f"onehot_feat{index}": (user_id + index) % 3
                for index in range(18)
            })
            user_rows.append(row)
        pd.DataFrame(user_rows).to_csv(
            cls.raw_dir / "user_features.csv", index=False
        )

        pd.DataFrame([
            {
                "video_id": video_id,
                "date": 20200630,
                "author_id": video_id % 3,
                "video_type": "NORMAL",
                "upload_type": "ShortImport",
                "video_duration": 5000 + video_id * 1000,
            }
            for video_id in range(5)
        ]).to_csv(cls.raw_dir / "item_daily_features.csv", index=False)
        pd.DataFrame({
            "video_id": range(5),
            "feat": ["[1, 2]", "[2]", "[3, 4]", "[]", "[1, 5]"],
        }).to_csv(cls.raw_dir / "item_categories.csv", index=False)

    def _validate_output(self, output, expected_counts):
        with open(output / "meta_data.json", "r", encoding="utf-8") as file:
            metadata = yaml.safe_load(file)
        self.assertEqual(metadata["sample_size"], expected_counts)

        for split in ("train", "valid", "test"):
            data_files = sorted((output / split / "data").glob("*.parquet"))
            self.assertTrue(data_files)
            for data_path in data_files:
                users = pd.read_parquet(
                    output / split / "user_info" / data_path.name
                ).set_index("user_index")
                items = pd.read_parquet(
                    output / split / "item_info" / data_path.name
                )
                data = pd.read_parquet(data_path)
                self.assertTrue(items.iloc[0].eq(0).all())
                self.assertTrue(set(data["high_watch"].unique()).issubset({0.0, 1.0}))
                for row in data.itertuples(index=False):
                    item_sequence = users.loc[row.user_index, "full_item_seq"]
                    action_sequence = users.loc[row.user_index, "full_action_seq"]
                    time_sequence = users.loc[
                        row.user_index, "full_timestamp_seq"
                    ]
                    self.assertEqual(item_sequence[row.seq_len], row.item_index)
                    self.assertEqual(len(item_sequence), len(action_sequence))
                    self.assertEqual(len(item_sequence), len(time_sequence))
                    self.assertEqual(list(time_sequence), sorted(time_sequence))

    def test_both_protocols_emit_valid_blocked_layouts(self):
        self._validate_output(
            self.big_output,
            {"total": 40, "train": 24, "valid": 8, "test": 8},
        )
        self._validate_output(
            self.dense_output,
            {"total": 46, "train": 32, "valid": 8, "test": 6},
        )

    def test_dense_protocol_keeps_small_events_out_of_train_history(self):
        train_lengths = []
        test_lengths = []
        for path in (self.dense_output / "train" / "user_info").glob("*.parquet"):
            train_lengths.extend(
                pd.read_parquet(path)["full_item_seq"].map(len).tolist()
            )
        for path in (self.dense_output / "test" / "user_info").glob("*.parquet"):
            test_lengths.extend(
                pd.read_parquet(path)["full_item_seq"].map(len).tolist()
            )
        self.assertEqual(max(train_lengths), 10)
        self.assertEqual(max(test_lengths), 13)

    def test_generated_config_loads_dataloader_and_qformer(self):
        with open(
            self.big_output / "dataset_config_snippet.yaml",
            "r",
            encoding="utf-8",
        ) as file:
            dataset_config = yaml.safe_load(file)["KuaiRec_Big_Watch_Action"]
        params = {
            **dataset_config,
            "dataset_id": "KuaiRec_Big_Watch_Action",
            "group_id": "user_id",
            "embedding_dim": 4,
            "verbose": 0,
            "pickle_feature_encoder": False,
        }
        processor = FeatureProcessor(**params)
        build_dataset(processor, **params)
        feature_map = FeatureMap(
            params["dataset_id"], str(self.big_output)
        )
        feature_map.load(str(self.big_output / "feature_map.json"), params)

        train_loader, _ = RankDataLoader(
            feature_map,
            stage="train",
            batch_size=4,
            shuffle=False,
            num_workers=0,
            max_len=4,
            drop_last=False,
            **params,
        ).make_iterator()
        batch = next(iter(train_loader))
        batch_dict, item_dict, mask, multi_masks = batch
        self.assertEqual(tuple(mask.shape), (4, 4))
        self.assertEqual(tuple(item_dict["item_id"].shape), (4, 5))
        self.assertEqual(len(multi_masks), 1)
        self.assertEqual(tuple(batch_dict["high_watch"].shape), (4,))

        model = QFormerCross3(
            feature_map,
            task="binary_classification",
            gpu=-1,
            embedding_dim=4,
            token_dim=8,
            num_heads=2,
            num_queries=2,
            num_ns_layers=1,
            num_unified_layers=1,
            ffn_ratio=2.0,
            max_len=4,
            num_tasks=1,
            tower_hidden_units=[8],
            loss=["binary_crossentropy"],
            dense_optimizer="AdamW",
            dense_learning_rate=1e-3,
            model_root=str(self.root / "checkpoints"),
            metrics=["AUC"],
            verbose=0,
            enable_torch_compile=False,
            enable_bf16=False,
        )
        prediction = model(batch)["high_watch_pred"]
        self.assertEqual(tuple(prediction.shape), (4, 1))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
