import unittest
from unittest.mock import patch

import torch

from model_zoo.QFormerCross10 import NonRecursiveCrossValueLayer
from model_zoo.QFormerCross11 import OriginalXQFormerStage
from model_zoo.QFormerCross12 import (
    OriginalXNSQFormerStage,
    QFormerCross12,
)
from tests.test_qformer_cross2 import DummyFeatureMap


def build_model():
    return QFormerCross12(
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
        model_root="/tmp/unirank-qformer12-test",
        metrics=["AUC"],
        verbose=0,
        enable_torch_compile=False,
        enable_bf16=False,
    )


def record_cross_kv(stage, queries, original_x, mask):
    seen = []
    patches = []
    for layer in stage.layers:
        original_forward = layer.cross_attn.forward

        def record(queries_arg, keys_arg, values_arg, mask_arg=None,
                   *, forward=original_forward):
            seen.append((keys_arg, values_arg))
            return forward(queries_arg, keys_arg, values_arg, mask_arg)

        patches.append(patch.object(
            layer.cross_attn, "forward", side_effect=record
        ))

    for current_patch in patches:
        current_patch.start()
    try:
        if queries is None:
            stage(original_x, mask)
        else:
            stage(queries, original_x, mask)
    finally:
        for current_patch in reversed(patches):
            current_patch.stop()
    return seen


class QFormerCross12Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.model = build_model()

    def test_both_stages_use_standard_qformer_layers(self):
        self.assertIsInstance(
            self.model.ns_qformer, OriginalXNSQFormerStage
        )
        self.assertIsInstance(
            self.model.unified_qformer, OriginalXQFormerStage
        )
        self.assertEqual(len(self.model.ns_qformer.layers), 2)
        self.assertEqual(len(self.model.unified_qformer.layers), 3)
        self.assertTrue(all(
            isinstance(layer, NonRecursiveCrossValueLayer)
            for layer in [
                *self.model.ns_qformer.layers,
                *self.model.unified_qformer.layers,
            ]
        ))

    def test_every_ns_layer_reuses_original_x_as_kv(self):
        original_x = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4)
        seen = record_cross_kv(
            self.model.ns_qformer, None, original_x, mask
        )
        self.assertEqual(len(seen), 2)
        for keys, values in seen:
            self.assertIs(keys, original_x)
            self.assertIs(values, original_x)

    def test_every_sequence_layer_reuses_original_x_as_kv(self):
        queries = torch.randn(2, 8, 8)
        original_x = torch.randn(2, 4, 8)
        mask = torch.tensor([
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ], dtype=torch.float32)
        seen = record_cross_kv(
            self.model.unified_qformer, queries, original_x, mask
        )
        self.assertEqual(len(seen), 3)
        for keys, values in seen:
            self.assertIs(keys, original_x)
            self.assertIs(values, original_x)

    def test_forward_backward_and_empty_history(self):
        batch_dict = {
            "user_id": torch.tensor([1, 2, 3]),
            "context": torch.tensor([2, 3, 4]),
        }
        item_dict = {
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
        mask = torch.tensor([
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ], dtype=torch.float32)
        output = self.model((batch_dict, item_dict, mask))
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


if __name__ == "__main__":
    unittest.main()
