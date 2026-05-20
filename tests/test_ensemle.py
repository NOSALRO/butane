import pytest
import torch

import butane


def model():
    return butane.nn.TimeMLP(
        input_dims=3,
        output_dims=2,
        hidden_dims=[128],
    )


def build_attachments(ensemble):
    def builder_optim(m, **kwargs):
        return torch.optim.Adam(m.parameters(), lr=1e-04, **kwargs)

    builder_ema = lambda m: butane.nn.EMA(m, decay=0.9999)
    ensemble.add_attachment("optimizer", builder_optim, weight_decay=1e-03)
    ensemble.add_attachment("ema", builder_ema)

def build_sequential_scheduler(optimizer, warm_start: int):
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1e-08,
        end_factor=1.0,
        total_iters=warm_start,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=1000,
        T_mult=1,
        eta_min=1e-06,
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[warm_start],
    )


@pytest.fixture
def ensemble():
    torch.manual_seed(32)
    return butane.nn.Ensemble(model, 3)


def test_ensemble_model(ensemble):
    assert ensemble is not None


def test_add_attachments(ensemble):
    build_attachments(ensemble)
    assert len(ensemble.attachments) > 0
    for k, v in ensemble.attachments.items():
        assert len(v) == ensemble.depth, "Attchments are not equal to model depth."

def test_add_lr_scheduler(ensemble):

    build_attachments(ensemble)
    ensemble.apply_on_attachment(
        name="lr_scheduler",
        attachment="optimizer",
        builder=build_sequential_scheduler,
        warm_start=100,
    )

def test_switch(ensemble):
    build_attachments(ensemble)
    model, attachments = ensemble.switch(2)
    try:
        ensemble.switch(8)
    except IndexError:
        assert True
    else:
        assert False, "Exception was not raised"

    assert "optimizer" in attachments, "Optimizer not in attachments."
    assert "ema" in attachments, "EMA not in attachments."


def test_device_switching(ensemble):
    build_attachments(ensemble)
    device = torch.device("cuda")
    ensemble.set_device(device)
    model, attachments = ensemble.switch(1)
    assert "cuda" in str(next(model.parameters()).device), "Model didnt switch device."
    pre_step_weights = torch.nn.utils.parameters_to_vector(model.parameters())
    x = torch.randn(1, 3).to(device)
    y = model(x)
    loss = y.pow(2).mean()
    attachments["optimizer"].zero_grad()
    loss.backward()
    attachments["optimizer"].step()
    after_step_weights = torch.nn.utils.parameters_to_vector(model.parameters())
    assert (pre_step_weights - after_step_weights).sum() != 0, "Model didnt update correctly."


def test_iter(ensemble):
    device = torch.device("cuda")
    build_attachments(ensemble)
    ensemble.set_device(device)
    x = torch.randn(1, 3).to(device)
    out = []
    for model, att in ensemble:
        out.append(model(x))
    assert len(out) == ensemble.depth, "Outputs are not equal to ensemble depth."
    for i in range(ensemble.depth):
        assert out[i] is not None, f"Output {i} is None."

    out = []
    ensemble.eval()
    for model, att in ensemble:
        out.append(model(x))
    assert len(out) == ensemble.depth, "Eval outputs are not equal to ensemble depth."
    for i in range(ensemble.depth):
        assert out[i] is not None, f"Eval output {i} is None."


def test_save_and_load_state_dict(ensemble):
    build_attachments(ensemble)
    device = torch.device("cpu")
    ensemble.set_device(device)

    model_orig, att_orig = ensemble.switch(0)
    x = torch.randn(1, 3)
    loss = model_orig(x).pow(2).mean()
    att_orig["optimizer"].zero_grad()
    loss.backward()
    att_orig["optimizer"].step()

    if hasattr(att_orig["ema"], "update"):
        att_orig["ema"].update()

    orig_params = torch.nn.utils.parameters_to_vector(model_orig.parameters())

    saved_state = ensemble.state_dict()
    assert "__ensemble_attachments__" in saved_state, "Attachments were not packed into state_dict."

    torch.manual_seed(999)
    new_ensemble = butane.nn.Ensemble(model, 3)
    build_attachments(new_ensemble)
    new_ensemble.set_device(device)

    model_new, att_new = new_ensemble.switch(0)
    new_params_before_load = torch.nn.utils.parameters_to_vector(model_new.parameters())
    assert not torch.allclose(orig_params, new_params_before_load), (
        "Fresh ensemble should have different weights before loading."
    )

    new_ensemble.load_state_dict(saved_state)

    model_loaded, att_loaded = new_ensemble.switch(0)
    loaded_params = torch.nn.utils.parameters_to_vector(model_loaded.parameters())

    assert torch.allclose(orig_params, loaded_params), (
        "Model weights did not match after load_state_dict."
    )

    orig_opt_state = att_orig["optimizer"].state_dict()
    loaded_opt_state = att_loaded["optimizer"].state_dict()

    assert len(orig_opt_state["state"]) > 0, (
        "Optimizer state should not be empty after taking a step."
    )
    assert len(orig_opt_state["state"]) == len(loaded_opt_state["state"]), (
        "Optimizer states did not load correctly."
    )

    orig_ema_state = (
        att_orig["ema"].state_dict() if hasattr(att_orig["ema"], "state_dict") else None
    )
    if orig_ema_state is not None:
        loaded_ema_state = att_loaded["ema"].state_dict()
        assert str(orig_ema_state) == str(loaded_ema_state), (
            "EMA states did not match after load_state_dict."
        )
