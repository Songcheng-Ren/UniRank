import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

from data.Taobao.preprocess_Taobao_seq_action import preprocess_and_split
from model_zoo.QFormerCross15 import QFormerCross15
from unirank.features import FeatureMap
from unirank.preprocess import FeatureProcessor, build_dataset
from unirank.pytorch.dataloaders import RankDataLoader


class TaobaoCtrPreprocessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="unirank-taobao-ctr-")
        cls.root = Path(cls.temp_dir.name)
        cls.raw_dir = cls.root / "raw"
        cls.raw_dir.mkdir()
        cls._write_raw_fixture()
        cls.output = cls.root / "Taobao_CTR"

        with contextlib.redirect_stdout(io.StringIO()):
            preprocess_and_split(
                data_dir=cls.raw_dir,
                output_dir=cls.output,
                task_mode="ctr",
                min_user_interactions=2,
                n_user_parts=2,
                chunk_size=7,
                buffer_flush_size=5,
                train_blocks=1,
                valid_blocks=1,
                test_blocks=1,
            )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @classmethod
    def _write_raw_fixture(cls):
        start = pd.Timestamp("2017-05-06 08:00:00", tz="Asia/Shanghai")
        rows = []
        for day in range(4):
            for user_id in range(4):
                for offset in range(2):
                    timestamp = start + pd.Timedelta(
                        days=day, minutes=user_id * 10 + offset
                    )
                    clicked = int((day + user_id + offset) % 3 == 0)
                    rows.append({
                        "user": user_id,
                        "adgroup_id": (day + user_id + offset) % 6,
                        "time_stamp": int(timestamp.timestamp()),
                        "pid": f"pid-{user_id % 2}",
                        "nonclk": 1 - clicked,
                        "clk": clicked,
                    })
        pd.DataFrame(rows).to_csv(cls.raw_dir / "raw_sample.csv", index=False)

        pd.DataFrame([
            {
                "adgroup_id": item_id,
                "cate_id": item_id % 3,
                "campaign_id": item_id % 2,
                "customer": item_id % 4,
                "brand": item_id % 3,
                "price": 10 + item_id,
            }
            for item_id in range(6)
        ]).to_csv(cls.raw_dir / "ad_feature.csv", index=False)

        user_rows = []
        for user_id in range(4):
            row = {"userid": user_id}
            row.update({
                "cms_segid": user_id % 2,
                "cms_group_id": user_id % 3,
                "final_gender_code": user_id % 2,
                "age_level": user_id % 3,
                "pvalue_level": user_id % 2,
                "shopping_level": user_id % 3,
                "occupation": user_id % 2,
                "new_user_class_level": user_id % 3,
            })
            user_rows.append(row)
        pd.DataFrame(user_rows).to_csv(
            cls.raw_dir / "user_profile.csv", index=False
        )

    def test_ctr_output_has_only_click_supervision(self):
        with open(self.output / "meta_data.json", encoding="utf-8") as file:
            meta = json.load(file)

        self.assertEqual(meta["task_mode"], "ctr")
        self.assertEqual(meta["label"], ["is_click"])
        self.assertEqual(meta["action_vocab"], {"exposure": 1, "click": 2})
        self.assertFalse(meta["behavior_log_usage"]["used"])
        self.assertEqual(meta["vocab_size"]["action"], 3)

        for split in ("train", "valid", "test"):
            data_path = next((self.output / split / "data").glob("*.parquet"))
            user_path = self.output / split / "user_info" / data_path.name
            data = pd.read_parquet(data_path)
            users = pd.read_parquet(user_path).set_index("user_index")
            self.assertIn("is_click", data.columns)
            self.assertTrue({"cart", "fav", "buy"}.isdisjoint(data.columns))
            self.assertTrue(set(data["is_click"].unique()).issubset({0.0, 1.0}))
            for row in data.itertuples(index=False):
                actions = users.loc[row.user_index, "full_action_seq"]
                self.assertTrue(set(actions).issubset({1, 2}))
                self.assertEqual(actions[row.seq_len], 2 if row.is_click else 1)

    def test_config_dataloader_and_qformer_accept_ctr_data(self):
        with open("config/dataset_config.yaml", encoding="utf-8") as file:
            dataset_config = yaml.safe_load(file)["Taobao_CTR"]
        with open(self.output / "meta_data.json", encoding="utf-8") as file:
            meta = json.load(file)

        dataset_config["data_root"] = str(self.root)
        for split in ("train", "valid", "test"):
            dataset_config[f"{split}_data"] = str(self.output / split / "data")
            dataset_config[f"{split}_user_info"] = str(
                self.output / split / "user_info"
            )
            dataset_config[f"{split}_item_info"] = str(
                self.output / split / "item_info"
            )
        for feature in dataset_config["feature_cols"]:
            name = feature["name"]
            if name in meta["vocab_size"] and meta["vocab_size"][name] > 0:
                feature["vocab_size"] = meta["vocab_size"][name]

        params = {
            **dataset_config,
            "dataset_id": "Taobao_CTR",
            "group_id": "user_id",
            "embedding_dim": 4,
            "verbose": 0,
            "pickle_feature_encoder": False,
        }
        processor = FeatureProcessor(**params)
        build_dataset(processor, **params)
        feature_map = FeatureMap("Taobao_CTR", str(self.output))
        feature_map.load(str(self.output / "feature_map.json"), params)

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
        batch_dict, _, mask, multi_masks = batch
        self.assertEqual(tuple(mask.shape), (4, 4))
        self.assertEqual(len(multi_masks), 1)
        self.assertEqual(tuple(batch_dict["is_click"].shape), (4,))

        model = QFormerCross15(
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
        prediction = model(batch)["is_click_pred"]
        self.assertEqual(tuple(prediction.shape), (4, 1))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
