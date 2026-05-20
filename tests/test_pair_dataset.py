import pytest
import torch

import butane


@pytest.fixture
def dataset():
    data = torch.arange(100).reshape(10, 10) + 1
    data_pair = torch.arange(100, 200).reshape(10, 10) + 1
    return butane.data.PairDataset(data=data, data_pair=data_pair)


@pytest.fixture
def dataset_targets():
    data = torch.arange(100).reshape(10, 10) + 1
    data_pair = torch.arange(100, 200).reshape(10, 10) + 1
    targets = torch.arange(10, dtype=torch.int32) + 1
    targets_pair = torch.arange(10, dtype=torch.int32) + 1
    return butane.data.PairDataset(
        data=data,
        data_pair=data_pair,
        targets=targets,
        targets_pair=targets_pair,
    )


def test_init(dataset):
    """Ensure initialization is correct"""
    assert dataset.data is not None, "data is not set properly."
    assert dataset.data_pair is not None, "data_pair is not set properly."


def test_init_w_targets(dataset_targets):
    """Ensure initialization is correct with targets"""
    assert dataset_targets.data is not None, "data is not set properly."
    assert dataset_targets.data_pair is not None, "data_pair is not set properly."
    assert dataset_targets.targets is not None, "targets is not set properly."
    assert dataset_targets.targets_pair is not None, "targets_pair is not set properly."


def test_sampling_w_targets(dataset_targets):
    """Verifies that sampling with targets is correct."""
    batch = dataset_targets[2]
    assert (batch["data"] == batch["targets"] - 100).all(), "int idx not working."

    batch = dataset_targets[:10]
    assert (batch["data"] == batch["targets"] - 100).all(), "slice idx not working."

    batch = dataset_targets[2:]
    assert (batch["data"] == batch["targets"] - 100).all(), "slice idx not working."

    batch = dataset_targets[::3]
    assert (batch["data"] == batch["targets"] - 100).all(), "slice idx not working."

    batch = dataset_targets[[3, 4, 5]]
    assert (batch["data"] == batch["targets"] - 100).all(), "list idx not working."

    batch = dataset_targets[torch.Tensor([3, 4, 5])]
    assert (batch["data"] == batch["targets"] - 100).all(), "tensor idx not working."

    batch = dataset_targets[tuple([3, 4, 5])]
    assert (batch["data"] == batch["targets"] - 100).all(), "tuple idx not working."

    batch = dataset_targets[::-1]
    assert (batch["data"] == batch["targets"] - 100).all(), "reverse step not working."


def test_sampling(dataset):
    """Verifies that sampling is correct."""
    batch = dataset[2]
    batch = dataset[:10]
    assert batch is not None, "slice idx not working."

    batch = dataset[2:]
    assert batch is not None, "slice idx not working."

    batch = dataset[::3]
    assert batch is not None, "slice idx not working."

    batch = dataset[[3, 4, 5]]
    assert batch is not None, "list idx not working."

    batch = dataset[torch.Tensor([3, 4, 5])]
    assert batch is not None, "tensor idx not working."

    batch = dataset[tuple([3, 4, 5])]
    assert batch is not None, "tuple idx not working."

    batch = dataset[::-1]
    assert batch is not None, "reverse step not working."


def test_device_movement(dataset_targets):
    """Verifies that all internal buffers move to the target device."""
    device = torch.device("cpu")  # Default
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    dataset_targets.to(device)

    assert dataset_targets.data.device.type == device.type
    assert dataset_targets.data_pair.device.type == device.type
    assert dataset_targets._start_idx.device.type == device.type
    assert dataset_targets._target_map_csr.device.type == device.type


def test_dataset_split(dataset_targets):
    """Verifies that splitting maintains pairing logic and class integrity."""
    original_len = len(dataset_targets)
    # Split 40% into the new dataset
    split_ds = dataset_targets.split(percentage=(1 - 0.4))

    assert len(dataset_targets) == 6
    assert len(split_ds) == 4

    # Verify that the new dataset is also a PairDataset
    assert isinstance(split_ds, butane.data.PairDataset)

    # Verify internal CSR maps were re-initialized for the new subset
    # (ref: butane _set_up_pairing must be called after split)
    assert split_ds._anchor_row_idx.size(0) == 4

    # Sampling should still work and respect class matching
    batch = split_ds[0]
    assert "data" in batch and "targets" in batch


def test_set_transforms(dataset_targets):
    """Verifies that vmap-based transforms are applied correctly."""

    def simple_transform(x):
        return x * 2.0

    dataset_targets.set_transforms(transforms=simple_transform)

    # Test batch fetch (uses vmap)
    batch = dataset_targets[:2]
    expected = (torch.arange(1, 21).reshape(2, 10).float()) * 2.0
    assert torch.allclose(batch["data"], expected)

    # Test scalar fetch (uses vmap + squeeze)
    single = dataset_targets[0]
    assert torch.allclose(single["data"], expected[0])


def test_flatten(dataset):
    """Verifies dimension flattening for MLP-based butane models."""
    dataset.flatten(dim=1)
    assert dataset.data.ndim == 2  # (10, 100)
    assert dataset.data_pair.ndim == 2


def test_return_tuple_mode(dataset_targets):
    """Verifies toggle between dictionary and tuple outputs."""
    dataset_targets._return_tuple = True
    output = dataset_targets[0]

    assert isinstance(output, tuple)
    assert len(output) == 2
    assert torch.is_tensor(output[0])  # data
    assert torch.is_tensor(output[1])  # data_pair


def test_deterministic_sampling(dataset_targets):
    """Ensures deterministic hashing produces consistent pairs across calls."""
    dataset_targets._deterministic = True

    fetch_1 = dataset_targets[0]
    fetch_2 = dataset_targets[0]

    assert torch.equal(fetch_1["targets"], fetch_2["targets"])


def test_set_data_runtime(dataset_targets):
    """Verifies that set() correctly updates data and re-triggers pairing."""
    new_data = torch.zeros_like(dataset_targets.data)
    dataset_targets.set(data=new_data)

    assert torch.all(dataset_targets.data == 0)
    # Check that pairing still functions (internal map should be valid)
    batch = dataset_targets[0]
    assert batch["data"].sum() == 0
