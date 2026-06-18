import pytest
import torch

import butane


# Helper function to generate inputs and dimensions systematically for DiT
def get_test_inputs(dims: int, hidden_dims: int = 64, batch_size: int = 2):
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

    t = torch.randn(batch_size, 1)
    
    # Pre-embedded sequence token context: [Batch, Sequence_Length, Hidden_Dims]
    sequence_ctx = torch.randn(batch_size, 4, hidden_dims)
    
    return input_dims, patch_size, x, t, sequence_ctx


def test_dit1d_time_conditioned_dit_mode():
    model = butane.nn.transformers.DiT1d(
        input_dims=[4, 16],
        patch_size=4,
        hidden_dims=64,
        depth=2,
        attention_heads=2,
    )
    x = torch.randn(2, 4, 16)
    t = torch.randn(2, 1)
    out = model(x, t=t)
    assert out.shape == x.shape


def test_dit1d_class_cfg():
    model = butane.nn.transformers.DiT1d(
        input_dims=[4, 16],
        patch_size=4,
        hidden_dims=64,
        depth=2,
        attention_heads=2,
        n_classes=10,
        class_drop_prob=0.1,
    )
    x = torch.randn(2, 4, 16)
    t = torch.randn(2, 1)

    # Test forward with active labels
    y = torch.randint(0, 10, (2,))
    out = model(x, t=t, labels=y)
    assert out.shape == x.shape

    # Test unconditional fallback branch via structural null class token
    out_uncond = model(x, t=t, labels=None)
    assert out_uncond.shape == x.shape


def test_dit2d_basic():
    model = butane.nn.transformers.DiT2d(
        input_dims=[3, 32, 32],
        patch_size=8,
        hidden_dims=128,
        depth=2,
        attention_heads=4,
    )
    x = torch.randn(2, 3, 32, 32)
    t = torch.randn(2, 1)
    out = model(x, t=t)
    assert out.shape == x.shape


def test_dit3d_basic():
    model = butane.nn.transformers.DiT3d(
        input_dims=[1, 8, 8, 8],
        patch_size=2,
        hidden_dims=64,
        depth=2,
        attention_heads=2,
    )
    x = torch.randn(2, 1, 8, 8, 8)
    t = torch.randn(2, 1)
    out = model(x, t=t)
    assert out.shape == x.shape


@pytest.mark.parametrize(
    "dit_cls, dims",
    [
        (butane.nn.transformers.DiT1d, 1),
        (butane.nn.transformers.DiT2d, 2),
        (butane.nn.transformers.DiT3d, 3),
    ],
)
class TestDiTConditioningVariations:
    
    def test_flat_vector_context_addition(self, dit_cls, dims):
        hidden_dims = 64
        input_dims, patch_size, x, t, _ = get_test_inputs(dims, hidden_dims=hidden_dims)
        
        model = dit_cls(
            input_dims=input_dims,
            patch_size=patch_size,
            hidden_dims=hidden_dims,
            depth=2,
            attention_heads=2,
            ctx_concat=False,
        )
        flat_ctx = torch.randn(x.shape[0], hidden_dims)

        kwargs = {"x": x, "ctx": flat_ctx}
        kwargs["t"] = t

        out = model(**kwargs)
        assert out.shape == x.shape

    def test_flat_vector_context_concatenation(self, dit_cls, dims):
        hidden_dims = 64
        input_dims, patch_size, x, t, _ = get_test_inputs(dims, hidden_dims=hidden_dims)
        ctx_dim = 16

        model = dit_cls(
            input_dims=input_dims,
            patch_size=patch_size,
            hidden_dims=hidden_dims,
            depth=2,
            attention_heads=2,
            ctx_concat=True,
            ctx_dim=ctx_dim,
        )
        flat_ctx = torch.randn(x.shape[0], ctx_dim)

        kwargs = {"x": x, "ctx": flat_ctx}
        kwargs["t"] = t

        out = model(**kwargs)
        assert out.shape == x.shape

    def test_cross_attention_token_conditioning(self, dit_cls, dims):
        hidden_dims = 64
        input_dims, patch_size, x, t, sequence_ctx = get_test_inputs(dims, hidden_dims=hidden_dims)

        model = dit_cls(
            input_dims=input_dims,
            patch_size=patch_size,
            hidden_dims=hidden_dims,
            depth=2,
            attention_heads=2,
            ctx_cross_attention=True,
            cross_attention_heads=2,
        )

        kwargs = {"x": x, "ctx": sequence_ctx}
        kwargs["t"] = t

        out = model(**kwargs)
        assert out.shape == x.shape

    def test_in_context_token_conditioning(self, dit_cls, dims):
        hidden_dims = 64
        input_dims, patch_size, x, t, sequence_ctx = get_test_inputs(dims, hidden_dims=hidden_dims)

        model = dit_cls(
            input_dims=input_dims,
            patch_size=patch_size,
            hidden_dims=hidden_dims,
            depth=2,
            attention_heads=2,
            ctx_in_context=True,
        )

        kwargs = {"x": x, "ctx": sequence_ctx}
        kwargs["t"] = t

        out = model(**kwargs)
        assert out.shape == x.shape


class TestDiTClassifierFreeGuidance:
    @pytest.mark.parametrize(
        "dit_cls, dims",
        [
            (butane.nn.transformers.DiT1d, 1),
            (butane.nn.transformers.DiT2d, 2),
            (butane.nn.transformers.DiT3d, 3),
        ],
    )
    def test_class_conditioning_and_cfg_dropout(self, dit_cls, dims):
        input_dims, patch_size, x, t, _ = get_test_inputs(dims)
        model = dit_cls(
            input_dims=input_dims,
            patch_size=patch_size,
            hidden_dims=64,
            depth=2,
            attention_heads=2,
            n_classes=10,
            class_drop_prob=0.2,
        )
        y = torch.randint(0, 10, (x.shape[0],))

        # Check train mode path (where random token drop is computed)
        model.train()
        out_train = model(x, t=t, labels=y)
        assert out_train.shape == x.shape

        # Check eval mode path
        model.eval()
        out_eval = model(x, t=t, labels=y)
        assert out_eval.shape == x.shape
