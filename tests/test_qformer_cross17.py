import unittest

import torch

from model_zoo import QFormerCross17
from model_zoo.QFormerCross8 import QFormerCross8
from tests.test_qformer_cross15_16 import model_kwargs
from tests.test_qformer_cross31 import make_inputs
from tests.test_qformer_cross2 import DummyFeatureMap


def build_model(output_dim=13, num_queries=8, gradient_checkpointing=False,
                num_blocks=1):
    kwargs = model_kwargs("/tmp/unirank-qformer17-test")
    kwargs["num_queries"] = num_queries
    kwargs["gradient_checkpointing"] = gradient_checkpointing
    return QFormerCross17(
        DummyFeatureMap(), output_dim=output_dim, num_blocks=num_blocks, **kwargs
    )


class QFormerCross17Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)

    def test_identity_projection_recovers_cross8_predictions(self):
        baseline = QFormerCross8(
            DummyFeatureMap(), **model_kwargs("/tmp/unirank-qformer8-test")
        ).eval()
        projected = build_model(output_dim=8 * 8).eval()
        copied_state = {}
        for name, value in baseline.state_dict().items():
            if name.startswith("ns_qformer."):
                name = "interaction_block." + name
            elif name.startswith("unified_qformer."):
                name = name.replace(
                    "unified_qformer.",
                    "interaction_block.sequence_qformer.",
                    1,
                )
            copied_state[name] = value
        result = projected.load_state_dict(copied_state, strict=False)
        self.assertEqual(set(result.missing_keys), {
            "interaction_block.output_projection.weight",
            "interaction_block.output_projection.bias",
        })
        self.assertEqual(result.unexpected_keys, [])
        with torch.no_grad():
            projection = projected.interaction_block.output_projection
            projection.weight.copy_(torch.eye(64))
            projection.bias.zero_()
            expected = baseline(make_inputs())
            actual = projected(make_inputs())
        for label in expected:
            torch.testing.assert_close(actual[label], expected[label])

    def test_fixed_output_dimension_and_backward_with_checkpointing(self):
        for checkpointing in (False, True):
            with self.subTest(gradient_checkpointing=checkpointing):
                model = build_model(gradient_checkpointing=checkpointing)
                observed_shapes = []
                handle = model.interaction_block.register_forward_hook(
                    lambda _, __, output: observed_shapes.append(output.shape)
                )
                output = model(make_inputs())
                handle.remove()
                self.assertEqual(observed_shapes, [torch.Size([3, 13])])
                self.assertEqual(set(output), {"click_pred", "buy_pred"})
                self.assertTrue(all(
                    prediction.shape == (3, 1)
                    and torch.isfinite(prediction).all()
                    for prediction in output.values()
                ))
                targets = [
                    torch.tensor([[1.0], [0.0], [1.0]]),
                    torch.tensor([[0.0], [1.0], [0.0]]),
                ]
                model.compute_loss(output, targets).backward()
                block = model.interaction_block
                for parameter in (
                    block.ns_qformer.learnable_queries,
                    block.sequence_qformer.layers[0].cross_attn.w_q.weight,
                    block.output_projection.weight,
                ):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                dense_ids = {id(parameter) for parameter in model._dense_params}
                self.assertIn(id(block.output_projection.weight), dense_ids)

    def test_padding_invariance_and_configurable_query_count(self):
        model = build_model(num_queries=3).eval()
        self.assertEqual(
            model.interaction_block.output_projection.in_features, 3 * 8
        )
        original_inputs = make_inputs()
        batch_dict, item_dict, mask = original_inputs
        changed_items = {
            name: values.clone() for name, values in item_dict.items()
        }
        for values in changed_items.values():
            history = values[:, :-1]
            history[~mask.bool()] = 1
        with torch.no_grad():
            expected = model(original_inputs)
            actual = model((batch_dict, changed_items, mask))
        for label in expected:
            torch.testing.assert_close(actual[label], expected[label])

    def test_rejects_invalid_output_dimension(self):
        for output_dim in (0, -1, 12.5):
            with self.subTest(output_dim=output_dim):
                with self.assertRaisesRegex(ValueError, "output_dim"):
                    build_model(output_dim=output_dim)

    def test_stacked_blocks_use_previous_output_and_original_sequence(self):
        for num_blocks in (2, 3):
            with self.subTest(num_blocks=num_blocks):
                model = build_model(num_blocks=num_blocks).eval()
                stack = model.interaction_block
                block_inputs = []
                transitions = []
                handles = [block.register_forward_pre_hook(
                    lambda _, args: block_inputs.append(args[:2])
                ) for block in stack.blocks]
                handles.extend(projection.register_forward_hook(
                    lambda _, __, output: transitions.append(output)
                ) for projection in stack.token_projections)
                try:
                    with torch.no_grad():
                        output = model(make_inputs())
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(len(block_inputs), num_blocks)
                self.assertEqual(len(transitions), num_blocks - 1)
                self.assertEqual(block_inputs[0][0].shape, (3, 4, 8))
                for index in range(1, num_blocks):
                    tokens, sequence = block_inputs[index]
                    self.assertEqual(tokens.shape, (3, 8, 8))
                    torch.testing.assert_close(
                        tokens, transitions[index - 1].reshape(3, 8, 8)
                    )
                    self.assertIs(sequence, block_inputs[0][1])
                self.assertTrue(all(
                    prediction.shape == (3, 1)
                    and torch.isfinite(prediction).all()
                    for prediction in output.values()
                ))
                parameter_ids = [
                    {id(parameter) for parameter in block.parameters()}
                    for block in stack.blocks
                ]
                for index in range(1, num_blocks):
                    self.assertTrue(parameter_ids[index].isdisjoint(
                        set().union(*parameter_ids[:index])
                    ))

    def test_stacked_backward_checkpointing_and_padding_invariance(self):
        for num_blocks in (2, 3):
            with self.subTest(num_blocks=num_blocks):
                model = build_model(
                    num_blocks=num_blocks, gradient_checkpointing=True
                )
                targets = [
                    torch.tensor([[1.0], [0.0], [1.0]]),
                    torch.tensor([[0.0], [1.0], [0.0]]),
                ]
                output = model(make_inputs())
                model.compute_loss(output, targets).backward()
                stack = model.interaction_block
                parameters = [
                    block.output_projection.weight for block in stack.blocks
                ] + [
                    projection.weight for projection in stack.token_projections
                ]
                dense_ids = {id(parameter) for parameter in model._dense_params}
                for parameter in parameters:
                    self.assertIn(id(parameter), dense_ids)
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

                model.eval()
                original_inputs = make_inputs()
                batch, items, mask = original_inputs
                changed_items = {
                    name: values.clone() for name, values in items.items()
                }
                for values in changed_items.values():
                    values[:, :-1][~mask.bool()] = 1
                with torch.no_grad():
                    expected = model(original_inputs)
                    actual = model((batch, changed_items, mask))
                for label in expected:
                    torch.testing.assert_close(actual[label], expected[label])

    def test_rejects_invalid_block_count(self):
        for num_blocks in (0, -1, 2.5):
            with self.subTest(num_blocks=num_blocks):
                with self.assertRaisesRegex(ValueError, "num_blocks"):
                    build_model(num_blocks=num_blocks)


if __name__ == "__main__":
    unittest.main()
