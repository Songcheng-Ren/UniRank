import unittest

import torch

from model_zoo.QFormerCross6 import QFormerCross6
from tests.test_qformer_cross2 import DummyFeatureMap


def build_model():
    return QFormerCross6(
        DummyFeatureMap(),
        task=["binary_classification", "binary_classification"],
        gpu=-1,
        embedding_dim=4,
        token_dim=8,
        num_heads=2,
        num_layers=2,
        ffn_ratio=2.0,
        qk_norm=True,
        max_len=4,
        num_tasks=2,
        tower_hidden_units=[16, 8],
        loss=["binary_crossentropy", "binary_crossentropy"],
        dense_optimizer="AdamW",
        dense_learning_rate=1e-3,
        model_root="/tmp/unirank-qformer6-test",
        metrics=["AUC"],
        verbose=0,
        enable_torch_compile=False,
        enable_bf16=False,
    )


class QFormerCross6Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
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

    def test_queries_are_ns_conditioned_and_kv_contains_ns_then_sequence(self):
        context = self.model.embedding_layer.embedding_layer(self.batch_dict)
        items = self.model.embedding_layer.embedding_layer(self.item_dict)
        ns_tokens = self.model._build_non_sequence_tokens(context, items)
        sequence_tokens = self.model._build_sequence_tokens(items, self.mask)
        queries = self.model._build_queries(ns_tokens, sequence_tokens, self.mask)
        kv_tokens, kv_mask = self.model._build_kv_tokens(
            ns_tokens, sequence_tokens, self.mask
        )

        self.assertEqual(self.model.num_queries, 8)
        self.assertEqual(queries.shape, (3, 8, 8))
        self.assertEqual(
            self.model.query_projection.in_features,
            (self.model.num_ns_tokens + 1) * self.model.token_dim,
        )
        self.assertEqual(
            self.model.query_projection.out_features,
            self.model.num_queries * self.model.token_dim,
        )
        self.assertEqual(
            kv_tokens.shape,
            (3, self.model.num_ns_tokens + self.mask.size(1), 8),
        )
        torch.testing.assert_close(
            kv_tokens[:, :self.model.num_ns_tokens], ns_tokens
        )
        torch.testing.assert_close(
            kv_tokens[:, self.model.num_ns_tokens:], sequence_tokens
        )
        torch.testing.assert_close(
            kv_mask[:, :self.model.num_ns_tokens],
            torch.ones_like(kv_mask[:, :self.model.num_ns_tokens]),
        )

    def test_forward_backward_and_padding_invariance(self):
        output = self.model((self.batch_dict, self.item_dict, self.mask))
        self.assertEqual(set(output), {"click_pred", "buy_pred"})
        self.assertTrue(all(
            torch.isfinite(prediction).all()
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
