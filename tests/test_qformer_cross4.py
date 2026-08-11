import unittest

from model_zoo.QFormerCross3 import SDPAQFormerLayer, SDPAQFormerStage
from model_zoo.QFormerCross4 import QFormerCross4
from tests.test_qformer_cross2 import DummyFeatureMap


class QFormerCross4Test(unittest.TestCase):
    def test_v4_defaults_to_sixteen_sdpa_queries(self):
        model = QFormerCross4(
            DummyFeatureMap(),
            task=["binary_classification", "binary_classification"],
            gpu=-1,
            embedding_dim=4,
            token_dim=8,
            num_heads=2,
            num_ns_layers=1,
            num_unified_layers=1,
            ffn_ratio=2.0,
            max_len=4,
            num_tasks=2,
            tower_hidden_units=[16, 8],
            loss=["binary_crossentropy", "binary_crossentropy"],
            dense_optimizer="AdamW",
            dense_learning_rate=1e-3,
            model_root="/tmp/unirank-qformer4-test",
            metrics=["AUC"],
            verbose=0,
            enable_torch_compile=False,
            enable_bf16=False,
        )

        self.assertEqual(model.num_queries, 16)
        self.assertEqual(tuple(model.ns_qformer.learnable_queries.shape), (16, 8))
        self.assertIsInstance(model.ns_qformer, SDPAQFormerStage)
        self.assertTrue(all(
            isinstance(layer, SDPAQFormerLayer)
            for layer in model.unified_qformer.layers
        ))


if __name__ == "__main__":
    unittest.main()
