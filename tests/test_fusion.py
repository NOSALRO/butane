import pytest
import torch

import butane


@pytest.mark.parametrize("fusion_type", ["additive", "film", "multiplicative"])
def test_conditional_fusion(fusion_type):
    emb = torch.randn(5, 128)
    features = torch.randn(5, 32)
    fuser = butane.nn.fusions.fusion_registry[fusion_type](128, 32)
    out = fuser(features, emb)
    assert out is not None
