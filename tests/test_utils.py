import unittest
import torch
import butane

class TestUtils(unittest.TestCase):

    def test_deterministic_sampling(self):
        # Setup data: 100x20x30 tensor
        # We will batch along dim=1 (size 20) with batch_size=10
        foo_data = torch.ones(100, 20, 30)
        foo_data[:, :10] *= 2  # First 10 indices along dim 1 are 2s
        foo_data[:, 10:] *= 3  # Next 10 indices along dim 1 are 3s

        batch_size = 10
        batching = butane.utils.batching(foo_data, batch_size, drop_last=False, dim=1)

        batches = list(batching)

        # Check number of batches (20 items / 10 batch_size = 2 batches)
        self.assertEqual(len(batches), 2)

        # Check Batch 1 Content (Should be all 2s)
        self.assertEqual(batches[0].shape, (100, 10, 30))
        self.assertTrue((batches[0] == 2).all())

        # Check Batch 2 Content (Should be all 3s)
        self.assertEqual(batches[1].shape, (100, 10, 30))
        self.assertTrue((batches[1] == 3).all())

    def test_tuple_handling(self):
        # Setup data: List of two tensors
        # Tensor A: 100x120x3 (batching dim 1, size 120)
        # Tensor B: 80x120x3 (batching dim 1, size 120)
        # Note: Inputs must typically have the same size along the batching dimension
        dim_size = 14  
        tensor_a = torch.randn(100, dim_size, 3)
        tensor_b = torch.randn(80, dim_size, 3)

        foo_data = [tensor_a, tensor_b]
        batch_size = 4

        # Test drop_last=False
        batching = butane.utils.batching(foo_data, batch_size, drop_last=False, dim=1)
        batches = list(batching)

        # Expected batches: ceil(14 / 4) = 4 batches (sizes: 4, 4, 4, 2)
        self.assertEqual(len(batches), 4)

        # Check the last partial batch size
        last_batch = batches[-1]
        self.assertEqual(last_batch[0].shape[1], 2)
        self.assertEqual(last_batch[1].shape[1], 2)

        # Verify reconstruction
        # Concatenate batches back along dim 1 and check against original
        recon_a = torch.cat([b[0] for b in batches], dim=1)
        recon_b = torch.cat([b[1] for b in batches], dim=1)

        self.assertTrue(torch.equal(recon_a, tensor_a))
        self.assertTrue(torch.equal(recon_b, tensor_b))

    def test_dict_handling(self):
        # Setup data: Dictionary of tensors
        # 'a': 100 items
        # 'b': 80 items
        # We batch along dim=0. The limit is determined by the shortest tensor (80).
        tensor_a = torch.ones(100, 2, 3) * 2
        tensor_b = torch.zeros(80, 2, 3)

        foo_data = dict(a=tensor_a, b=tensor_b)
        batch_size = 7

        batching = butane.utils.batching(foo_data, batch_size, drop_last=False, dim=0)
        batches = list(batching)

        # Expected limit is 80 (from tensor_b). 
        # Batches: ceil(80 / 7) = 12 batches.
        # Last batch size: 80 % 7 = 3.
        self.assertEqual(len(batches), 12)

        # Check first batch properties
        self.assertIsInstance(batches[0], dict)
        self.assertIn('a', batches[0])
        self.assertIn('b', batches[0])
        self.assertEqual(batches[0]['a'].shape[0], 7)
        self.assertTrue((batches[0]['a'] == 2).all())
        self.assertTrue((batches[0]['b'] == 0).all())

        # Check last batch size
        last_batch = batches[-1]
        self.assertEqual(last_batch['a'].shape[0], 3)
        self.assertEqual(last_batch['b'].shape[0], 3)

if __name__ == '__main__':
    unittest.main()
