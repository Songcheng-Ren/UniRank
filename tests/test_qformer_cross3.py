import unittest

import torch

from model_zoo.QFormerCross3 import (
    QFormerCross3,
    SDPACrossValueAttention,
    SDPAQFormerLayer,
    SDPAQFormerStage,
)
from tests.test_qformer_cross2 import DummyFeatureMap


def unfused_head_local_cross_value(module, queries, keys, mask=None):
    batch_size, num_queries, _ = queries.shape
    sequence_length = keys.size(1)
    q = module.w_q(queries).view(
        batch_size, num_queries, module.num_heads, module.head_dim
    ).transpose(1, 2)
    k = module.w_k(keys).view(
        batch_size, sequence_length, module.num_heads, module.head_dim
    ).transpose(1, 2)
    v = module.w_fi(keys).view(
        batch_size, sequence_length, module.num_heads, module.head_dim
    ).transpose(1, 2)
    q = module.q_norm(q)
    k = module.k_norm(k)

    scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale
    if mask is not None:
        valid = mask.to(dtype=torch.bool).view(
            batch_size, 1, 1, sequence_length
        )
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    attention = torch.softmax(scores, dim=-1)
    if mask is not None:
        attention = attention * valid.to(dtype=attention.dtype)
        attention = attention / attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
    attended_features = torch.matmul(attention, v)

    q_interaction = module.w_qi(queries).view(
        batch_size, num_queries, module.num_heads, module.head_dim
    ).transpose(1, 2)
    pair_summary = q_interaction * attended_features
    pair_summary = pair_summary.transpose(1, 2).contiguous().view(
        batch_size, num_queries, module.token_dim
    )
    return module.w_o(module.w_v_pair(pair_summary))


def build_model():
    return QFormerCross3(
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
        model_root="/tmp/unirank-qformer3-test",
        metrics=["AUC"],
        verbose=0,
        enable_torch_compile=False,
        enable_bf16=False,
    )


class QFormerCross3Test(unittest.TestCase):
    def test_v3_uses_independent_sdpa_primitives(self):
        model = build_model()
        self.assertIsInstance(model.ns_qformer, SDPAQFormerStage)
        self.assertTrue(all(
            isinstance(layer, SDPAQFormerLayer)
            for layer in model.unified_qformer.layers
        ))

    def test_sdpa_cross_value_matches_unfused_reference(self):
        torch.manual_seed(31)
        module = SDPACrossValueAttention(12, 3, qk_norm=True).double()
        queries = torch.randn(2, 4, 12, dtype=torch.double, requires_grad=True)
        keys = torch.randn(2, 5, 12, dtype=torch.double, requires_grad=True)
        mask = torch.tensor([
            [1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
        ], dtype=torch.double)

        expected = unfused_head_local_cross_value(
            module, queries, keys, mask
        )
        actual = module(queries, keys, keys, mask)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertTrue(torch.isfinite(keys.grad).all())

    def test_v3_forward_backward_and_empty_history(self):
        torch.manual_seed(37)
        model = build_model()
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

        output = model((batch_dict, item_dict, mask))
        self.assertEqual(set(output), {"click_pred", "buy_pred"})
        self.assertTrue(all(
            torch.isfinite(prediction).all()
            for prediction in output.values()
        ))
        sum(prediction.mean() for prediction in output.values()).backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
