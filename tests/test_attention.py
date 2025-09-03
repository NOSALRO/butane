import unittest
import torch
import butane

class TestAttention(unittest.TestCase):

    def test_self_attention(self):
        self_att = butane.nn.SelfAttention(256, n_heads=1)
        self.assertIsNotNone(self_att(torch.randn(2, 3, 256)))

        local_self_att = butane.nn.LocalSelfAttention2d(3, n_heads=1)
        self.assertIsNotNone(local_self_att(torch.randn(2, 3, 256, 256)))

    def test_cross_attention(self):
        cross_att = butane.nn.CrossAttention(256, n_heads=1)
        self.assertIsNotNone(cross_att(torch.randn(2, 3, 256), torch.randn(2, 3, 256)))

        local_cross_att = butane.nn.LocalCrossAttention2d(3, n_heads=1)
        self.assertIsNotNone(local_cross_att(torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)))

if __name__ == '__main__':
    unittest.main()
