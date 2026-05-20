import h5py
import numpy as np
import pytest
import torch

from butane.data import TrajectoryDataset


@pytest.fixture
def list_dataset():
    """Generates a dataset from a list of variable-length 1D tensors."""
    data = []
    for _ in range(10):
        ep_len = torch.randint(20, 100, size=(1,)).item()
        data.append(torch.arange(ep_len).unsqueeze(-1).float())

    return TrajectoryDataset(data=data, horizon=16, history=2, align_start=True)


@pytest.fixture
def dict_dataset():
    data_dict = {"obs": [], "actions": []}
    for _ in range(10):
        ep_len = torch.randint(20, 100, size=(1,)).item()
        data_dict["obs"].append(torch.randn(ep_len, 3, 64, 64))  # Fake images
        data_dict["actions"].append(torch.randn(ep_len, 7))  # Fake joint actions

    return TrajectoryDataset(data=data_dict, horizon=8, history=4, align_start=False)

def test_list_initialization_and_shapes(list_dataset):
    assert len(list_dataset) > 0, "Dataset length should be greater than 0"

    sample = list_dataset[10]

    assert "data" in sample, "Missing 'data' root key"
    assert "targets" in sample, "Missing 'targets' root key"

    data_seq, target_seq = sample

    assert data_seq.shape[0] == 2, "History length mismatch"
    assert target_seq.shape[0] == 16, "Horizon length mismatch"


def test_dict_initialization_and_shapes(dict_dataset):
    sample = dict_dataset[10]

    assert "data" in sample, "Missing 'data' root key"
    assert "targets" in sample, "Missing 'targets' root key"

    assert "obs" in sample["data"]
    assert "actions" in sample["targets"]

    assert sample["data"]["obs"].shape[0] == 4
    assert sample["targets"]["actions"].shape[0] == 8


def test_padding_edge_cases(dict_dataset):
    sample = dict_dataset[0]

    obs_history = sample["data"]["obs"]

    assert obs_history.shape[0] == 4

    assert torch.allclose(obs_history[0], obs_history[1]), "Left padding failed to replicate"


def test_h5_ingestion(tmp_path):
    fpath = tmp_path / "dummy_trajectory.h5"
    with h5py.File(fpath, "w") as f:
        for i in range(3):
            traj = f.create_group(f"traj_{i}")
            traj.create_dataset("obs", data=np.random.randn(50, 10))  # 50 steps, 10 dim obs
            traj.create_dataset("actions", data=np.random.randn(50, 4))  # 50 steps, 4 dim actions

    f = h5py.File(fpath, "r")

    parsed_data = {"obs": [], "actions": []}
    for k in f.keys():
        traj_data = load_h5_data(f[k])
        parsed_data["obs"].append(traj_data["obs"])
        parsed_data["actions"].append(traj_data["actions"])

    dataset = TrajectoryDataset(data=parsed_data, horizon=10, history=3)
    assert len(dataset) == 150

    sample = dataset[75]  # Grab a random middle step
    assert sample["data"]["obs"].shape[0] == 3
    assert sample["targets"]["actions"].shape[0] == 10
