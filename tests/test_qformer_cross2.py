import unittest
from collections import OrderedDict

import torch

from model_zoo.QFormerCross2 import QFormerCross2


class DummyFeatureMap:
    def __init__(self):
        self.dataset_id = "QFormerCross2Test"
        self.data_dir = "/tmp"
        self.labels = ["click", "buy"]
        self.group_id = None
        self.features = OrderedDict([
            ("user_id", {
                "type": "categorical", "vocab_size": 17, "embedding_dim": 4,
            }),
            ("context", {
                "type": "categorical", "vocab_size": 7, "embedding_dim": 3,
            }),
            ("item_id", {
                "type": "categorical", "vocab_size": 19, "embedding_dim": 5,
                "source": "item",
            }),
            ("category", {
                "type": "categorical", "vocab_size": 11, "embedding_dim": 3,
                "source": "item",
            }),
            ("action", {
                "type": "categorical", "vocab_size": 5, "embedding_dim": 2,
                "source": "action",
            }),
        ])


def build_model():
    return QFormerCross2(
        DummyFeatureMap(),
        task=["binary_classification", "binary_classification"],
        gpu=-1,
        embedding_dim=4,
        token_dim=8,
        num_heads=2,
        num_queries=3,
        num_ns_layers=1,
        num_unified_layers=2,
        ffn_ratio=2.0,
        qk_norm=True,
        max_len=4,
        num_tasks=2,
        tower_hidden_units=[16, 8],
        loss=["binary_crossentropy", "binary_crossentropy"],
        dense_optimizer="AdamW",
        dense_learning_rate=1e-3,
        model_root="/tmp/unirank-qformer2-test",
        metrics=["AUC"],
        verbose=0,
        enable_torch_compile=False,
        enable_bf16=False,
    )


class QFormerCross2Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = build_model()
        self.batch_dict = {
            "user_id": torch.tensor([1, 2, 3]),
            "context": torch.tensor([2, 3, 4]),
        }
        self.item_dict = {
            "item_id": torch.tensor([
                [3, 4, 5, 6, 7],
                [2, 3, 4, 5, 6],
                [8, 9, 10, 11, 12],
            ]),
            "category": torch.tensor([
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
            ]),
            "action": torch.tensor([
                [1, 2, 3, 4, 0],
                [2, 3, 4, 1, 0],
                [3, 4, 1, 2, 0],
            ]),
        }
        self.mask = torch.tensor([
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ], dtype=torch.float32)

    def test_architecture_has_no_group_or_multi_embedding_path(self):
        self.assertEqual(self.model.ns_features, [
            "user_id", "context", "item_id", "category",
        ])
        self.assertEqual(self.model.sequence_features, [
            "item_id", "category", "action",
        ])
        self.assertEqual(self.model.ns_field_embedding.shape, (4, 8))
        self.assertFalse(hasattr(self.model, "group_qformers"))
        self.assertFalse(hasattr(self.model, "user_multi_proj"))

    def test_forward_and_backward_are_finite(self):
        output = self.model((self.batch_dict, self.item_dict, self.mask))
        self.assertEqual(set(output), {"click_pred", "buy_pred"})
        for prediction in output.values():
            self.assertEqual(prediction.shape, (3, 1))
            self.assertTrue(torch.isfinite(prediction).all())

        sum(prediction.mean() for prediction in output.values()).backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        ))

    def test_padding_values_do_not_change_predictions(self):
        self.model.eval()
        changed_item_dict = {
            feature: values.clone()
            for feature, values in self.item_dict.items()
        }
        for values in changed_item_dict.values():
            values[0, :2] = 1
            values[2, :4] = 1

        with torch.no_grad():
            expected = self.model((self.batch_dict, self.item_dict, self.mask))
            actual = self.model((self.batch_dict, changed_item_dict, self.mask))
        for label in expected:
            torch.testing.assert_close(
                actual[label], expected[label], atol=1e-6, rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
