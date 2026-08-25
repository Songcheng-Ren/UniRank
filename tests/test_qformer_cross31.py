import unittest

import torch

from model_zoo.QFormerCross3 import QFormerCross3, SDPAQFormerLayer
from model_zoo.QFormerCross31 import (
    FieldAwareSDPACrossValueAttention,
    FieldAwareSDPAQFormerLayer,
    FieldAwareSDPAQFormerStage,
    QFormerCross31,
)
from tests.test_qformer_cross2 import DummyFeatureMap


def build_model(model_cls):
    return model_cls(
        DummyFeatureMap(),
        task=["binary_classification", "binary_classification"],
        gpu=-1,
        embedding_dim=4,
        token_dim=8,
        num_heads=2,
        num_queries=3,
        num_ns_layers=2,
        num_unified_layers=2,
        ffn_ratio=2.0,
        qk_norm=True,
        max_len=4,
        num_tasks=2,
        tower_hidden_units=[16, 8],
        loss=["binary_crossentropy", "binary_crossentropy"],
        dense_optimizer="AdamW",
        dense_learning_rate=1e-3,
        model_root="/tmp/unirank-qformer31-test",
        metrics=["AUC"],
        verbose=0,
        enable_torch_compile=False,
        enable_bf16=False,
    )


def make_inputs():
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
    return batch_dict, item_dict, mask


def unfused_field_aware_cross_value(
        module, queries, keys, field_head_log_scale, mask=None):
    batch_size, num_queries, _ = queries.shape
    num_fields = keys.size(1)
    q = module.w_q(queries).view(
        batch_size, num_queries, module.num_heads, module.head_dim
    ).transpose(1, 2)
    k = module.w_k(keys).view(
        batch_size, num_fields, module.num_heads, module.head_dim
    ).transpose(1, 2)
    v = module.w_fi(keys).view(
        batch_size, num_fields, module.num_heads, module.head_dim
    ).transpose(1, 2)
    q = module.q_norm(q)
    k = module.k_norm(k)

    scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale
    field_scale = field_head_log_scale.exp().transpose(0, 1)
    scores = scores * field_scale.view(1, module.num_heads, 1, num_fields)
    if mask is not None:
        valid = mask.to(dtype=torch.bool).view(
            batch_size, 1, 1, num_fields
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


class QFormerCross31Test(unittest.TestCase):
    def test_nonzero_scale_matches_field_head_logit_formula(self):
        torch.manual_seed(61)
        module = FieldAwareSDPACrossValueAttention(
            12, 3, qk_norm=True
        ).double()
        queries = torch.randn(2, 4, 12, dtype=torch.double)
        fields = torch.randn(2, 5, 12, dtype=torch.double)
        alpha = torch.tensor([
            [0.0, 0.2, -0.1],
            [0.3, -0.2, 0.1],
            [-0.4, 0.5, 0.0],
            [0.1, 0.0, 0.4],
            [0.2, -0.3, 0.3],
        ], dtype=torch.double, requires_grad=True)
        mask = torch.tensor([
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ], dtype=torch.double)

        expected = unfused_field_aware_cross_value(
            module, queries, fields, alpha, mask
        )
        actual = module(queries, fields, fields, alpha, mask)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        actual.square().mean().backward()
        self.assertIsNotNone(alpha.grad)
        self.assertTrue(torch.isfinite(alpha.grad).all())

    def test_v31_is_v3_plus_one_shared_f_by_h_parameter(self):
        torch.manual_seed(67)
        baseline = build_model(QFormerCross3)
        torch.manual_seed(67)
        calibrated = build_model(QFormerCross31)

        load_result = calibrated.load_state_dict(
            baseline.state_dict(), strict=False
        )
        self.assertEqual(
            load_result.missing_keys,
            ["ns_qformer.field_head_log_scale"],
        )
        self.assertEqual(load_result.unexpected_keys, [])
        self.assertEqual(calibrated.ns_qformer.field_names, (
            "user_id", "context", "item_id", "category",
        ))
        self.assertEqual(
            tuple(calibrated.ns_qformer.field_head_log_scale.shape),
            (4, 2),
        )
        torch.testing.assert_close(
            calibrated.ns_qformer.field_head_log_scale,
            torch.zeros(4, 2),
        )
        num_baseline_params = sum(
            parameter.numel() for parameter in baseline.parameters()
        )
        num_calibrated_params = sum(
            parameter.numel() for parameter in calibrated.parameters()
        )
        self.assertEqual(num_calibrated_params - num_baseline_params, 4 * 2)

        self.assertIsInstance(
            calibrated.ns_qformer, FieldAwareSDPAQFormerStage
        )
        self.assertTrue(all(
            isinstance(layer, FieldAwareSDPAQFormerLayer)
            for layer in calibrated.ns_qformer.layers
        ))
        self.assertTrue(all(
            isinstance(layer, SDPAQFormerLayer)
            for layer in calibrated.unified_qformer.layers
        ))
        self.assertFalse(any(
            "field_head_log_scale" in name
            for name, _ in calibrated.unified_qformer.named_parameters()
        ))

        inputs = make_inputs()
        baseline.eval()
        calibrated.eval()
        with torch.no_grad():
            baseline_output = baseline(inputs)
            calibrated_output = calibrated(inputs)
        for label in baseline_output:
            torch.testing.assert_close(
                calibrated_output[label], baseline_output[label]
            )

    def test_v31_forward_backward_updates_alpha(self):
        torch.manual_seed(71)
        model = build_model(QFormerCross31)
        output = model(make_inputs())
        sum(prediction.mean() for prediction in output.values()).backward()
        alpha_grad = model.ns_qformer.field_head_log_scale.grad
        self.assertIsNotNone(alpha_grad)
        self.assertTrue(torch.isfinite(alpha_grad).all())


if __name__ == "__main__":
    unittest.main()

