import pytest
import torch

import butane


@pytest.fixture
def model_config():
    return dict(
        input_dims=[3, 32, 32],
        hidden_dims=768,
        mlp_ratio=4,
        depth=8,
        cfg_drop_prob=0.8,
    )


@pytest.fixture
def model(model_config):
    return butane.nn.DiT2d(**model_config)


@pytest.fixture
def model_conditional_2d(model_config):
    model_config["condition_input_dims"] = [1, 32, 32]
    model_config["condition_in_context"] = True
    return butane.nn.DiT2d(**model_config)


@pytest.fixture
def model_class_conditional(model_config):
    model_config["n_classes"] = 10
    model_config["cfg_drop_prob"] = 0.8
    model_config["condition_in_context"] = False
    return butane.nn.DiT2d(**model_config)


@pytest.fixture
def model_conditional_1d(model_config):
    model_config["condition_input_dims"] = [1, 32]
    model_config["condition_in_context"] = True
    return butane.nn.DiT2d(**model_config)


@pytest.fixture
def model_multimodal_condition_dict(model_config):
    # model_config["condition_input_dims"] = ((1, 32, 32), (1, 32))
    model_config["condition_input_dims"] = dict(img=(1, 32, 32), state=(1, 32))
    model_config["condition_patch_size"] = dict(img=(8, 8), state=(8,))
    model_config["condition_in_context"] = True
    return butane.nn.DiT2d(**model_config)


@pytest.fixture
def model_multimodal_condition_tuple(model_config):
    model_config["condition_input_dims"] = ((1, 32, 32), (1, 32))
    model_config["condition_in_context"] = True
    model_config["condition_patch_size"] = ((8, 8), (8,))
    return butane.nn.DiT2d(**model_config)


def test_model_init(model, model_config):
    assert model is not None, "Model is not initialized"


def test_uncoditional_forward(model, model_config):
    x = torch.randn((1, *model_config.get("input_dims")))
    t = torch.rand((1, 1))
    out = model(x, t)
    assert tuple(out.shape) == tuple(x.shape), f"Expected {x.shape}, got {out.shape}"


def test_conditional_2d_forward(model_conditional_2d, model_config):
    x = torch.randn((1, *model_config.get("input_dims")))
    c = torch.randn((1, *model_config.get("condition_input_dims")))
    t = torch.rand((1, 1))
    out = model_conditional_2d(x, t, c)
    assert tuple(out.shape) == tuple(x.shape), f"Expected {x.shape}, got {out.shape}"


def test_conditional_1d_forward(model_conditional_1d, model_config):
    x = torch.randn((1, *model_config.get("input_dims")))
    c = torch.randn((1, *model_config.get("condition_input_dims")))
    t = torch.rand((1, 1))
    out = model_conditional_1d(x, t, c)
    assert tuple(out.shape) == tuple(x.shape), f"Expected {x.shape}, got {out.shape}"


def test_class_conditional_forward(model_class_conditional, model_config):
    x = torch.randn((1, *model_config.get("input_dims")))
    c = torch.randint(0, 10, size=(1,))
    t = torch.rand((1, 1))
    out = model_class_conditional(x, t, c=c)
    assert tuple(out.shape) == tuple(x.shape), f"Expected {x.shape}, got {out.shape}"


def test_multimodal_tuple_forward(model_multimodal_condition_tuple, model_config):
    x = torch.randn((1, *model_config.get("input_dims")))
    condition_input_dims = model_config.get("condition_input_dims")
    condition_patch_size = ((8, 8), (8,))
    c = (torch.randn((1, *condition_input_dims[0])), torch.randn((1, *condition_input_dims[1])))
    t = torch.rand((1, 1))
    out = model_multimodal_condition_tuple(x, t, c)
    assert tuple(out.shape) == tuple(x.shape), f"Expected {x.shape}, got {out.shape}"


def test_multimodal_dict_forward(model_multimodal_condition_dict, model_config):
    x = torch.randn((1, *model_config.get("input_dims")))
    condition_input_dims = model_config.get("condition_input_dims")
    condition_patch_size = dict(img=(8, 8), state=(8,))
    c = dict(
        img=torch.randn((1, *condition_input_dims["img"])),
        state=torch.randn((1, *condition_input_dims["state"])),
    )
    t = torch.rand((1, 1))
    out = model_multimodal_condition_dict(x, t, c)
    assert tuple(out.shape) == tuple(x.shape), f"Expected {x.shape}, got {out.shape}"
