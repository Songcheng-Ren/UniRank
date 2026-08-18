import unittest
from unittest.mock import patch

import torch

from model_zoo.QFormerCross8 import RecursiveCrossValueLayer
from model_zoo.QFormerCross10 import (
    NonRecursiveCrossValueLayer,
    NonRecursiveCrossValueStage,
)
from model_zoo.QFormerCross11 import OriginalXQFormerStage, QFormerCross11
from tests.test_qformer_cross2 import DummyFeatureMap


def build_model():
    return QFormerCross11(
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
        model_root="/tmp/unirank-qformer11-test",
        metrics=["AUC"],
        verbose=0,
        enable_torch_compile=False,
        enable_bf16=False,
    )


class QFormerCross11Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
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

    def test_both_stages_restore_standard_qformer_layers(self):
        self.assertIsInstance(
            self.model.ns_qformer, NonRecursiveCrossValueStage
        )
        self.assertIsInstance(
            self.model.unified_qformer, OriginalXQFormerStage
        )
        self.assertEqual(len(self.model.ns_qformer.layers), 2)
        self.assertEqual(len(self.model.unified_qformer.layers), 3)
        for layer in [
            *self.model.ns_qformer.layers,
            *self.model.unified_qformer.layers,
        ]:
            self.assertIsInstance(layer, NonRecursiveCrossValueLayer)
            self.assertNotIsInstance(layer, RecursiveCrossValueLayer)
            self.assertTrue(hasattr(layer, "self_attn"))
            self.assertTrue(hasattr(layer, "cross_attn"))
            self.assertTrue(hasattr(layer, "ffn"))

    def test_every_sequence_cross_attention_reuses_original_x_as_kv(self):
        stage = self.model.unified_qformer
        queries = torch.randn(2, 8, 8)
        original_x = torch.randn(2, 4, 8)
        mask = torch.tensor([
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ], dtype=torch.float32)
        seen = []
        patches = []
        for layer in stage.layers:
            original_forward = layer.cross_attn.forward

            def record(queries_arg, keys_arg, values_arg, mask_arg=None,
                       *, forward=original_forward):
                seen.append((keys_arg, values_arg))
                return forward(
                    queries_arg, keys_arg, values_arg, mask_arg
                )

            patches.append(patch.object(
                layer.cross_attn, "forward", side_effect=record
            ))

        for current_patch in patches:
            current_patch.start()
        try:
            stage(queries, original_x, mask)
        finally:
            for current_patch in reversed(patches):
                current_patch.stop()

        self.assertEqual(len(seen), 3)
        for keys, values in seen:
            self.assertIs(keys, original_x)
            self.assertIs(values, original_x)

    def test_each_layer_runs_self_then_cross_then_ffn(self):
        layer = self.model.unified_qformer.layers[0]
        call_order = []
        hooks = [
            layer.self_attn.register_forward_pre_hook(
                lambda *_: call_order.append("self_attn")
            ),
            layer.cross_attn.register_forward_pre_hook(
                lambda *_: call_order.append("cross_attn")
            ),
            layer.ffn.register_forward_pre_hook(
                lambda *_: call_order.append("ffn")
            ),
        ]
        try:
            layer(
                torch.randn(2, 8, 8),
                torch.randn(2, 4, 8),
                torch.ones(2, 4),
            )
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(
            call_order, ["self_attn", "cross_attn", "ffn"]
        )

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
            expected = self.model((
                self.batch_dict, self.item_dict, self.mask
            ))
            actual = self.model((
                self.batch_dict, changed_item_dict, self.mask
            ))
        for label in expected:
            torch.testing.assert_close(
                actual[label], expected[label], atol=1e-6, rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
