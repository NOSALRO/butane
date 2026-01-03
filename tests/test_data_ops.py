import unittest
import torch
import butane

class TestDataOps(unittest.TestCase):

    def test_data_replication(self):
        fpath = "data/cifar100/cifar100_test_data.pt"
        fpath_targets = "data/cifar100/cifar100_test_targets.pt"
        cifar100_data = torch.load(fpath, weights_only=True).to(dtype=torch.uint8)
        targets = torch.tensor(torch.load(fpath_targets), dtype=torch.int32)

        unique_targets, counts = torch.unique(targets, return_counts=True)
        self.assertTrue((counts == counts[0]).all())
        initial_count = counts[0]

        targets = targets.repeat_interleave(dim=0, repeats=5)
        unique_targets, counts = torch.unique(targets, return_counts=True)
        self.assertTrue((counts == initial_count * 5).all())

    def test_data_drop_to_size(self):
        fpath = "data/cifar100/cifar100_test_data.pt"
        fpath_targets = "data/cifar100/cifar100_test_targets.pt"
        cifar100_data = torch.load(fpath, weights_only=True).to(dtype=torch.uint8)
        targets = torch.tensor(torch.load(fpath_targets), dtype=torch.int32)
        data_1, targets_1 = butane.data.ops.drop_to_max_size(cifar100_data, max_size=1000, respect_targets=True, y=targets, seed=0)
        target_counts = torch.unique(targets_1, return_counts=True)[1]
        self.assertTrue((target_counts == target_counts[0]).all())
        data_2, targets_2 = butane.data.ops.drop_to_max_size(cifar100_data, max_size=1000, respect_targets=True, y=targets, seed=0)
        self.assertTrue((data_1 == data_2).all())

        data_1, targets_1 = butane.data.ops.drop_to_max_size(cifar100_data, max_size=1000, respect_targets=False, y=targets, seed=3)
        target_counts = torch.unique(targets_1, return_counts=True)[1]
        self.assertTrue(not (target_counts == target_counts[0]).all())
        data_2, targets_2 = butane.data.ops.drop_to_max_size(cifar100_data, max_size=1000, respect_targets=False, y=targets, seed=2)
        self.assertTrue(not (data_1 == data_2).all())

    def test_data_randperm(self):
        fpath = "data/cifar100/cifar100_test_data.pt"
        fpath_targets = "data/cifar100/cifar100_test_targets.pt"
        cifar100_data = torch.load(fpath, weights_only=True).to(dtype=torch.uint8)
        targets = torch.tensor(torch.load(fpath_targets), dtype=torch.int32)

        indices = butane.data.ops.randperm(len(cifar100_data), respect_targets=True, y=targets, seed=0)
        data_1, targets_1 = cifar100_data[indices[:1000]], targets[indices[:1000]]
        target_counts = torch.unique(targets_1, return_counts=True)[1]
        self.assertTrue((target_counts == target_counts[0]).all())
        indices = butane.data.ops.randperm(len(cifar100_data), respect_targets=True, y=targets, seed=0)
        data_2, targets_2 = cifar100_data[indices[:1000]], targets[indices[:1000]]
        self.assertTrue((data_1 == data_2).all())

if __name__ == "__main__":
    unittest.main()

