import unittest

import torch

from model_zoo.QFormerCross8 import (
    RecursiveCrossValueLayer,
    RecursiveCrossValueStage,
    RecursiveSequenceCrossValueStage,
)
from model_zoo.QFormerCross9 import QFormerCross9
from tests.test_qformer_cross2 import DummyFeatureMap


class QFormerCross9Test(unittest.TestCase):
    def test_v9_defaults_both_crossvalue_stacks_to_three_layers(self):
        model = QFormerCross9(
            DummyFeatureMap(),
            task=["binary_classification", "binary_classification"],
            gpu=-1,
            embedding_dim=4,
            token_dim=8,
            num_heads=2,
            num_queries=8,
            num_unified_layers=3,
            ffn_ratio=2.0,
            qk_norm=True,
            max_len=4,
            num_tasks=2,
            tower_hidden_units=[16, 8],
            loss=["binary_crossentropy", "binary_crossentropy"],
            dense_optimizer="AdamW",
            dense_learning_rate=1e-3,
            model_root="/tmp/unirank-qformer9-test",
            metrics=["AUC"],
            verbose=0,
            enable_torch_compile=False,
            enable_bf16=False,
        )

        self.assertEqual(model.num_queries, 8)
        self.assertIsInstance(model.ns_qformer, RecursiveCrossValueStage)
        self.assertIsInstance(
            model.unified_qformer, RecursiveSequenceCrossValueStage
        )
        self.assertEqual(len(model.ns_qformer.layers), 3)
        self.assertEqual(len(model.unified_qformer.layers), 3)
        self.assertTrue(all(
            isinstance(layer, RecursiveCrossValueLayer)
            for layer in [
                *model.ns_qformer.layers,
                *model.unified_qformer.layers,
            ]
        ))

        batch_dict = {
            "user_id": torch.tensor([1, 2]),
            "context": torch.tensor([2, 3]),
        }
        item_dict = {
            "item_id": torch.tensor([
                [3, 4, 5, 6, 7],
                [2, 3, 4, 5, 6],
            ]),
            "category": torch.tensor([
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
            ]),
            "action": torch.tensor([
                [1, 2, 3, 4, 0],
                [2, 3, 4, 1, 0],
            ]),
        }
        mask = torch.tensor([
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ], dtype=torch.float32)
        output = model((batch_dict, item_dict, mask))
        self.assertTrue(all(
            prediction.shape == (2, 1)
            and torch.isfinite(prediction).all()
            for prediction in output.values()
        ))
        sum(prediction.mean() for prediction in output.values()).backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
