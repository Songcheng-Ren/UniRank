import unittest

import torch
import torch.nn.functional as F

from model_zoo.LoopCTR import LoopBlock, LoopCTR, SwiGLU
from tests.test_qformer_cross2 import DummyFeatureMap


def build_model(num_loops=3, inference_loops=1):
    return LoopCTR(
        DummyFeatureMap(),
        task=["binary_classification", "binary_classification"],
        gpu=-1,
        embedding_dim=4,
        token_dim=8,
        num_heads=2,
        num_loops=num_loops,
        inference_loops=inference_loops,
        ffn_ratio=2.0,
        qk_norm=False,
        max_len=4,
        num_tasks=2,
        tower_hidden_units=[16, 8],
        loss=["binary_crossentropy", "binary_crossentropy"],
        dense_optimizer="AdamW",
        dense_learning_rate=1e-3,
        model_root="/tmp/unirank-loopctr-test",
        metrics=["AUC"],
        verbose=0,
        enable_torch_compile=False,
        enable_bf16=False,
    )


class LoopCTRTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
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

    @property
    def inputs(self):
        return self.batch_dict, self.item_dict, self.mask

    def test_architecture_has_one_shared_loop_and_swiglu(self):
        self.assertIsInstance(self.model.loop_block, LoopBlock)
        self.assertIsInstance(self.model.loop_block.ffn, SwiGLU)
        self.assertFalse(hasattr(self.model, "loop_blocks"))
        self.assertFalse(any(
            "expert" in name.lower() or "hcr" in name.lower()
            for name, _ in self.model.named_modules()
        ))

        zero_loop_model = build_model(num_loops=0, inference_loops=0)
        parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        zero_loop_parameter_count = sum(
            parameter.numel() for parameter in zero_loop_model.parameters()
        )
        self.assertEqual(parameter_count, zero_loop_parameter_count)

    def test_training_returns_every_depth_and_process_loss_is_mean(self):
        self.model.train()
        output = self.model(self.inputs)
        labels = self.model.feature_map.labels
        expected_keys = {f"{label}_pred" for label in labels}
        expected_keys.update(
            f"{label}_pred_loop_{depth}"
            for label in labels
            for depth in range(self.model.num_loops + 1)
        )
        self.assertEqual(set(output), expected_keys)

        y_true = [
            torch.tensor([[1.0], [0.0], [1.0]]),
            torch.tensor([[0.0], [1.0], [0.0]]),
        ]
        actual = self.model.compute_loss(output, y_true)
        manual_depth_losses = []
        for depth in range(self.model.num_loops + 1):
            manual_depth_losses.append(sum(
                F.binary_cross_entropy(
                    output[f"{label}_pred_loop_{depth}"], y_true[index]
                )
                for index, label in enumerate(labels)
            ))
        expected = torch.stack(manual_depth_losses).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in self.model.parameters()
        ))

    def test_eval_executes_configured_number_of_shared_loops(self):
        self.model.eval()
        calls = []
        handle = self.model.loop_block.register_forward_hook(
            lambda *_: calls.append(1)
        )
        try:
            with torch.no_grad():
                output = self.model(self.inputs)
            self.assertEqual(len(calls), 1)
            self.assertEqual(set(output), {"click_pred", "buy_pred"})

            calls.clear()
            self.model.set_inference_loops(0)
            with torch.no_grad():
                self.model(self.inputs)
            self.assertEqual(len(calls), 0)

            calls.clear()
            self.model.set_inference_loops(3)
            with torch.no_grad():
                self.model(self.inputs)
            self.assertEqual(len(calls), 3)
        finally:
            handle.remove()

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
            expected = self.model(self.inputs)
            actual = self.model((
                self.batch_dict, changed_item_dict, self.mask
            ))
        for label in expected:
            torch.testing.assert_close(
                actual[label], expected[label], atol=1e-6, rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
