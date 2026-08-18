import unittest

import torch

from model_zoo.QFormerCross3 import SDPACrossValueAttention
from model_zoo.QFormerCross7 import (
    RecursiveCrossValueLayer,
    RecursiveCrossValueStage,
)
from model_zoo.QFormerCross8 import (
    QFormerCross8,
    RecursiveSequenceCrossValueStage,
)
from tests.test_qformer_cross2 import DummyFeatureMap


def build_model():
    return QFormerCross8(
        DummyFeatureMap(),
        task=["binary_classification", "binary_classification"],
        gpu=-1,
        embedding_dim=4,
        token_dim=8,
        num_heads=2,
        num_queries=8,
        num_ns_layers=2,
        num_unified_layers=3,
        ffn_ratio=2.0,
        qk_norm=True,
        max_len=4,
        num_tasks=2,
        tower_hidden_units=[16, 8],
        loss=["binary_crossentropy", "binary_crossentropy"],
        dense_optimizer="AdamW",
        dense_learning_rate=1e-3,
        model_root="/tmp/unirank-qformer8-test",
        metrics=["AUC"],
        verbose=0,
        enable_torch_compile=False,
        enable_bf16=False,
    )


class QFormerCross8Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
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

    def test_both_stages_use_recursive_crossvalue(self):
        self.assertEqual(self.model.num_queries, 8)
        self.assertIsInstance(
            self.model.ns_qformer, RecursiveCrossValueStage
        )
        self.assertIsInstance(
            self.model.unified_qformer,
            RecursiveSequenceCrossValueStage,
        )
        self.assertEqual(len(self.model.ns_qformer.layers), 2)
        self.assertEqual(len(self.model.unified_qformer.layers), 3)
        for layer in [
            *self.model.ns_qformer.layers,
            *self.model.unified_qformer.layers,
        ]:
            self.assertIsInstance(layer, RecursiveCrossValueLayer)
            self.assertIsInstance(
                layer.cross_attn, SDPACrossValueAttention
            )
            self.assertFalse(hasattr(layer, "self_attn"))
            self.assertFalse(hasattr(layer, "ffn"))

    def test_sequence_stage_matches_manual_recurrence_before_post_processing(self):
        stage = self.model.unified_qformer
        queries = torch.randn(2, 8, 8)
        sequence = torch.randn(2, 4, 8)
        mask = torch.tensor([
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ], dtype=torch.float32)

        expected = queries
        for layer in stage.layers:
            expected = layer(expected, sequence, mask)
        mixed = stage.post_self_attn(expected, expected, expected)
        expected = stage.post_self_norm(expected + stage.dropout(mixed))
        ffn_output = stage.post_ffn(expected)
        expected = stage.post_ffn_norm(
            expected + stage.dropout(ffn_output)
        )
        actual = stage(queries, sequence, mask)
        torch.testing.assert_close(actual, expected)

    def test_forward_backward_empty_history_and_padding_invariance(self):
        output = self.model((self.batch_dict, self.item_dict, self.mask))
        self.assertEqual(set(output), {"click_pred", "buy_pred"})
        self.assertTrue(all(
            prediction.shape == (3, 1)
            and torch.isfinite(prediction).all()
            for prediction in output.values()
        ))
        sum(prediction.mean() for prediction in output.values()).backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        ))

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
