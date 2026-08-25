import unittest

import torch

from model_zoo.QFormerCross8 import (
    QFormerCross8,
    RecursiveSequenceCrossValueStage,
)
from model_zoo.QFormerCross81 import (
    FieldAwareMaskedSDPAMultiHeadAttention,
    FieldAwareRecursiveCrossValueLayer,
    FieldAwareRecursiveCrossValueStage,
    QFormerCross81,
)
from tests.test_qformer_cross31 import build_model, make_inputs


class QFormerCross81Test(unittest.TestCase):
    def test_v81_is_v8_plus_one_shared_f_by_h_parameter(self):
        torch.manual_seed(73)
        baseline = build_model(QFormerCross8)
        torch.manual_seed(73)
        calibrated = build_model(QFormerCross81)

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
            calibrated.ns_qformer, FieldAwareRecursiveCrossValueStage
        )
        self.assertIsInstance(
            calibrated.ns_qformer.initial_read,
            FieldAwareMaskedSDPAMultiHeadAttention,
        )
        self.assertTrue(all(
            isinstance(layer, FieldAwareRecursiveCrossValueLayer)
            for layer in calibrated.ns_qformer.layers
        ))
        self.assertIsInstance(
            calibrated.unified_qformer,
            RecursiveSequenceCrossValueStage,
        )
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

    def test_v81_forward_backward_updates_shared_alpha(self):
        torch.manual_seed(79)
        model = build_model(QFormerCross81)
        output = model(make_inputs())
        sum(prediction.mean() for prediction in output.values()).backward()
        alpha_grad = model.ns_qformer.field_head_log_scale.grad
        self.assertIsNotNone(alpha_grad)
        self.assertTrue(torch.isfinite(alpha_grad).all())


if __name__ == "__main__":
    unittest.main()

