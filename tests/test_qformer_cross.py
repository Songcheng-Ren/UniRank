import unittest
from collections import OrderedDict

import torch

from model_zoo.QFormerCross import CrossValueAttention, QFormerCross, QFormerStage


def naive_cross_value_attention(module, queries, keys, mask=None):
    """Unfused reference for the head-local SDPA CrossValue path."""
    batch_size, num_queries, _ = queries.shape
    seq_len = keys.size(1)

    q = module.w_q(queries).view(
        batch_size, num_queries, module.num_heads, module.head_dim
    ).transpose(1, 2)
    k = module.w_k(keys).view(
        batch_size, seq_len, module.num_heads, module.head_dim
    ).transpose(1, 2)
    q = module.q_norm(q)
    k = module.k_norm(k)
    v = module.w_fi(keys).view(
        batch_size, seq_len, module.num_heads, module.head_dim
    ).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale
    if mask is not None:
        key_mask = mask.to(dtype=torch.bool).view(batch_size, 1, 1, seq_len)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores, dim=-1)
    if mask is not None:
        attn = attn * key_mask.to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    attended_features = torch.matmul(attn, v)
    q_inter = module.w_qi(queries).view(
        batch_size, num_queries, module.num_heads, module.head_dim
    ).transpose(1, 2)
    pair_summary = q_inter * attended_features
    pair_summary = pair_summary.transpose(1, 2).contiguous().view(
        batch_size, num_queries, module.token_dim
    )
    return module.w_o(module.w_v_pair(pair_summary))


class DummyFeatureMap:
    def __init__(self):
        self.dataset_id = "QFormerCrossTest"
        self.data_dir = "/tmp"
        self.labels = ["click", "like"]
        self.group_id = None
        self.features = OrderedDict([
            ("user_id", {
                "type": "categorical", "vocab_size": 17, "embedding_dim": 4,
            }),
            ("context", {
                "type": "categorical", "vocab_size": 7, "embedding_dim": 4,
            }),
            ("item_a", {
                "type": "categorical", "vocab_size": 19, "embedding_dim": 4,
                "source": "item",
            }),
            ("item_b", {
                "type": "categorical", "vocab_size": 13, "embedding_dim": 6,
                "source": "item",
            }),
            ("action", {
                "type": "categorical", "vocab_size": 5, "embedding_dim": 2,
                "source": "action",
            }),
        ])


class QFormerCrossTest(unittest.TestCase):
    def test_group_bounds_are_embedding_offsets(self):
        self.assertEqual(
            QFormerCross._compute_group_bounds([16] * 7, 4),
            [0, 32, 64, 96, 112],
        )
        self.assertEqual(
            QFormerCross._compute_group_bounds([4, 6, 2], 2),
            [0, 10, 12],
        )

    def test_masked_query_conditioning_ignores_padding(self):
        torch.manual_seed(7)
        stage = QFormerStage(
            token_dim=8,
            num_heads=2,
            num_layers=2,
            num_queries=3,
            ffn_dim=16,
            qk_norm=True,
        ).eval()
        valid_features = torch.randn(2, 3, 8)
        padded_features = torch.cat([valid_features, torch.randn(2, 4, 8)], dim=1)
        valid_mask = torch.ones(2, 3)
        padded_mask = torch.cat([valid_mask, torch.zeros(2, 4)], dim=1)

        expected = stage(valid_features, valid_mask)
        actual = stage(padded_features, padded_mask)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_sdpa_cross_value_attention_matches_unfused_reference(self):
        torch.manual_seed(11)
        attention = CrossValueAttention(12, 3, qk_norm=True).double()
        queries = torch.randn(2, 4, 12, dtype=torch.double, requires_grad=True)
        keys = torch.randn(2, 5, 12, dtype=torch.double, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.double)

        expected = naive_cross_value_attention(attention, queries, keys, mask)
        actual = attention(queries, keys, keys, mask)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(queries.grad).all())
        self.assertTrue(torch.isfinite(keys.grad).all())

    def test_end_to_end_forward_and_backward(self):
        torch.manual_seed(17)
        feature_map = DummyFeatureMap()
        model = QFormerCross(
            feature_map,
            task=["binary_classification", "binary_classification"],
            gpu=-1,
            embedding_dim=4,
            token_dim=8,
            num_heads=2,
            num_layers=1,
            num_groups=2,
            num_queries_per_group=2,
            seq_num_queries=2,
            ffn_ratio=2.0,
            qk_norm=True,
            use_target_aware_shortcut=True,
            num_tasks=2,
            tower_hidden_units=[16, 8],
            loss=["binary_crossentropy", "binary_crossentropy"],
            dense_optimizer="AdamW",
            dense_learning_rate=1e-3,
            model_root="/tmp/unirank-qformer-test",
            metrics=["AUC"],
            verbose=0,
            enable_torch_compile=False,
            enable_bf16=False,
            gradient_checkpointing=True,
        )
        batch_size, history_len = 3, 4
        batch_dict = {
            "user_id": torch.randint(0, 17, (batch_size,)),
            "context": torch.randint(0, 7, (batch_size,)),
        }
        item_dict = {
            "item_a": torch.randint(0, 19, (batch_size, history_len + 1)),
            "item_b": torch.randint(0, 13, (batch_size, history_len + 1)),
            "action": torch.randint(0, 5, (batch_size, history_len + 1)),
        }
        mask = torch.tensor([
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ], dtype=torch.float32)

        output = model((batch_dict, item_dict, mask))
        self.assertEqual(set(output), {"click_pred", "like_pred"})
        for prediction in output.values():
            self.assertEqual(prediction.shape, (batch_size, 1))
            self.assertTrue(torch.isfinite(prediction).all())
        sum(prediction.mean() for prediction in output.values()).backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
