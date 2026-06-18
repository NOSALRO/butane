import pytest
import torch

import butane


# Helper function to generate trajectory sequence tensors [Batch, State_Dim, Horizon]
def get_trajectory_test_inputs(
    batch_size: int = 2, state_dim: int = 14, horizon: int = 20, hidden_dims: int = 64
):
    input_dims = [state_dim, horizon]
    x = torch.randn(batch_size, state_dim, horizon)

    # Multi-modal context stream (e.g., target goal representations, vision token features)
    sequence_ctx = torch.randn(batch_size, 5, hidden_dims)

    return input_dims, x, sequence_ctx


def test_transformer1d_unconditional_trajectory_forward():
    # Example input: 14 states/actions evaluated over a 20 step trajectory window
    input_dims, x, _ = get_trajectory_test_inputs(state_dim=14, horizon=20)

    model = butane.nn.transformers.Transformer1d(
        input_dims=input_dims,
        hidden_dims=64,
        depth=2,
        attention_heads=2,
    )

    # Confirm patch size was hardcoded to 1 to enforce token-per-step trajectory mapping
    assert model._patch_size == (1,)

    out = model(x)
    assert out.shape == x.shape


def test_transformer1d_time_dependent_enforcement():
    input_dims, x, _ = get_trajectory_test_inputs()
    model = butane.nn.transformers.Transformer1d(input_dims=input_dims)
    t = torch.randn(x.shape[0], 1)

    with pytest.raises(ValueError, match="Model is configured as `time_dependent=False`"):
        model(x, t=t)


class TestTransformer1dTokenConditioning:
    def test_cross_attention_conditioning(self):
        hidden_dims = 64
        input_dims, x, sequence_ctx = get_trajectory_test_inputs(hidden_dims=hidden_dims)
        ctx_dim = sequence_ctx.shape[-1]

        model = butane.nn.transformers.Transformer1d(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            depth=2,
            attention_heads=2,
            ctx_cross_attention=True,
            cross_attention_heads=2,
            ctx_dim=ctx_dim,
        )

        out = model(x, ctx=sequence_ctx)
        assert out.shape == x.shape

    def test_in_context_prefix_conditioning(self):
        hidden_dims = 64
        input_dims, x, sequence_ctx = get_trajectory_test_inputs(hidden_dims=hidden_dims)

        model = butane.nn.transformers.Transformer1d(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            depth=2,
            attention_heads=2,
            ctx_in_context=True,
        )

        out = model(x, ctx=sequence_ctx)
        assert out.shape == x.shape
