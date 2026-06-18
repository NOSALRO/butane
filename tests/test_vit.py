import pytest
import torch

import butane


# Helper function to generate inputs and dimensions systematically for ViT
def get_vit_test_inputs(dims: int, hidden_dims: int = 64, batch_size: int = 2):
    if dims == 1:
        input_dims = [4, 16]
        patch_size = 4
        x = torch.randn(batch_size, 4, 16)
    elif dims == 2:
        input_dims = [3, 16, 16]
        patch_size = 4
        x = torch.randn(batch_size, 3, 16, 16)
    elif dims == 3:
        input_dims = [1, 8, 8, 8]
        patch_size = 2
        x = torch.randn(batch_size, 1, 8, 8, 8)
    else:
        raise ValueError(f"Invalid dimension rank: {dims}")

    # Pre-embedded multi-modal token context sequence: [Batch, Sequence_Length, Hidden_Dims]
    sequence_ctx = torch.randn(batch_size, 4, hidden_dims)
    
    return input_dims, patch_size, x, sequence_ctx


@pytest.mark.parametrize(
    "vit_cls, dims",
    [
        (butane.nn.transformers.ViT1d, 1),
        (butane.nn.transformers.ViT2d, 2),
        (butane.nn.transformers.ViT3d, 3),
    ],
)
def test_vit_unconditional_forward(vit_cls, dims):
    input_dims, patch_size, x, _ = get_vit_test_inputs(dims)
    model = vit_cls(
        input_dims=input_dims,
        patch_size=patch_size,
        hidden_dims=64,
        depth=2,
        attention_heads=2,
    )
    
    # Ensure time embedding sub-modules were bypassed to save parameters
    assert not hasattr(model, "time_embedder")
    assert not hasattr(model, "embedding_projection")
    
    out = model(x)
    assert out.shape == x.shape


@pytest.mark.parametrize(
    "vit_cls, dims",
    [
        (butane.nn.transformers.ViT1d, 1),
        (butane.nn.transformers.ViT2d, 2),
        (butane.nn.transformers.ViT3d, 3),
    ],
)
def test_vit_time_dependent_enforcement(vit_cls, dims):
    input_dims, patch_size, x, _ = get_vit_test_inputs(dims)
    model = vit_cls(
        input_dims=input_dims,
        patch_size=patch_size,
    )
    t = torch.randn(x.shape[0], 1)
    
    # Passing a timestep to a ViT must raise a ValueError matching UNet/DiT base rules
    with pytest.raises(ValueError, match="Model is configured as `time_dependent=False`"):
        model(x, t=t)


@pytest.mark.parametrize(
    "vit_cls, dims",
    [
        (butane.nn.transformers.ViT1d, 1),
        (butane.nn.transformers.ViT2d, 2),
        (butane.nn.transformers.ViT3d, 3),
    ],
)
class TestViTTokenConditioning:

    def test_cross_attention_token_conditioning(self, vit_cls, dims):
        hidden_dims = 64
        input_dims, patch_size, x, sequence_ctx = get_vit_test_inputs(dims, hidden_dims=hidden_dims)
        ctx_dim = sequence_ctx.shape[-1]

        model = vit_cls(
            input_dims=input_dims,
            patch_size=patch_size,
            hidden_dims=hidden_dims,
            depth=2,
            attention_heads=2,
            ctx_cross_attention=True,
            cross_attention_heads=2,
            ctx_dim=ctx_dim,
        )

        out = model(x, ctx=sequence_ctx)
        assert out.shape == x.shape

    def test_in_context_token_conditioning(self, vit_cls, dims):
        hidden_dims = 64
        input_dims, patch_size, x, sequence_ctx = get_vit_test_inputs(dims, hidden_dims=hidden_dims)

        model = vit_cls(
            input_dims=input_dims,
            patch_size=patch_size,
            hidden_dims=hidden_dims,
            depth=2,
            attention_heads=2,
            ctx_in_context=True,
        )

        out = model(x, ctx=sequence_ctx)
        assert out.shape == x.shape
