import unittest

import torch

from model_zoo.QFormerCross3 import QFormerCross3
from model_zoo.QFormerCross8 import QFormerCross8
from model_zoo.QFormerCross15 import QFormerCross15
from model_zoo.QFormerCross16 import QFormerCross16
from tests.test_qformer_cross2 import DummyFeatureMap


def model_kwargs(model_root):
    return {
        "task": ["binary_classification", "binary_classification"],
        "gpu": -1,
        "embedding_dim": 4,
        "token_dim": 8,
        "num_heads": 2,
        "num_queries": 8,
        "num_ns_layers": 2,
        "num_unified_layers": 3,
        "ffn_ratio": 2.0,
        "qk_norm": True,
        "max_len": 4,
        "num_tasks": 2,
        "tower_hidden_units": [16, 8],
        "loss": ["binary_crossentropy", "binary_crossentropy"],
        "dense_optimizer": "AdamW",
        "dense_learning_rate": 1e-3,
        "model_root": model_root,
        "metrics": ["AUC"],
        "verbose": 0,
        "enable_torch_compile": False,
        "enable_bf16": False,
    }


class QFormerCross15And16Test(unittest.TestCase):
    def setUp(self):
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

    def assert_rms_qk_variant(self, copy_cls, version):
        copied = copy_cls(
            DummyFeatureMap(),
            **model_kwargs(f"/tmp/unirank-qformer{version}-copy-test"),
        ).eval()

        qk_norms = [
            module
            for name, module in copied.named_modules()
            if name.endswith(("q_norm", "k_norm"))
        ]
        self.assertTrue(qk_norms)
        self.assertTrue(all(
            isinstance(module, torch.nn.RMSNorm)
            for module in qk_norms
        ))
        output = copied((self.batch_dict, self.item_dict, self.mask))
        self.assertTrue(all(
            prediction.shape == (3, 1)
            and torch.isfinite(prediction).all()
            for prediction in output.values()
        ))
        sum(prediction.mean() for prediction in output.values()).backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in copied.parameters()
        ))

    def test_v15_is_v3_copy_with_rms_qk_norm(self):
        self.assertNotIn(QFormerCross3, QFormerCross15.__mro__)
        self.assert_rms_qk_variant(QFormerCross15, 15)

    def test_v16_is_v8_copy_with_rms_qk_norm(self):
        self.assertNotIn(QFormerCross8, QFormerCross16.__mro__)
        self.assert_rms_qk_variant(QFormerCross16, 16)


if __name__ == "__main__":
    unittest.main()
