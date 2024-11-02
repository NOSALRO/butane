import unittest
import torch
import butane

conv_config = {
    "channels": [32, 32],
    "activation_function": [torch.nn.ReLU()],
    "conv_kernels": [3, 2],
    "conv_stride": [1, 1],
    "conv_pad": [0, 1],
    "conv_bias": [True, False],
    "conv_pad_mode": ['zeros', 'reflect'],
    "pool_kernels": [2, 3],
    "pool_stride": [1, 2],
    "pool_pad": [1, 1],
    "dropout": [.3],
    "output_activation": True,
    "normalization": [False, True],
}

conv_transpose_config = {
    "channels": [32, 32],
    "activation_function": [torch.nn.ReLU()],
    "conv_kernels": [3, 2],
    "conv_stride": [2, 1],
    "conv_pad": [0, 1],
    "conv_bias": [True, False],
    "conv_output_padding": [1, 0],
    "dropout": [.3],
    "output_activation": True,
    "normalization": [False, True],
}

conv_upsample_config = {
    "channels": [32, 32],
    "activation_function": [torch.nn.ReLU()],
    "conv_kernels": [3, 2],
    "conv_stride": [2, 1],
    "conv_pad": [0, 1],
    "conv_bias": [True, False],
    "conv_pad_mode": ['zeros', 'reflect'],
    "upsample_scale_factor": [2, 1.5],
    "upsample_mode": ["nearest"],
    "upsample_align_corners": [False, True],
    "dropout": [.3],
    "output_activation": True,
    "normalization": [False, True],
}


class TestConvBlocks(unittest.TestCase):

    def test_conv_1d_block(self):
        self.assertTrue(butane.nn.Conv1dBlock(
            input_dims = [1, 28],
            pool = [torch.nn.MaxPool1d, torch.nn.AvgPool1d],
            normalization_type = [torch.nn.BatchNorm1d, torch.nn.LayerNorm],
            **conv_config
        ))

    def test_conv_2d_block(self):
        self.assertTrue(butane.nn.Conv2dBlock(
            input_dims = [1, 28, 28],
            pool = [torch.nn.MaxPool2d, torch.nn.AvgPool2d],
            normalization_type = [torch.nn.BatchNorm2d, torch.nn.LayerNorm],
            **conv_config
        ))

    def test_conv_3d_block(self):
        self.assertTrue(butane.nn.Conv3dBlock(
            input_dims = [1, 28, 28, 28],
            pool = [torch.nn.MaxPool3d, torch.nn.AvgPool3d],
            normalization_type = [torch.nn.BatchNorm3d, torch.nn.LayerNorm],
            **conv_config
        ))

    def test_conv_transpose_1d_block(self):
        self.assertTrue(butane.nn.ConvTranspose1dBlock(
            input_dims = [1, 28],
            normalization_type = [torch.nn.BatchNorm1d, torch.nn.LayerNorm],
            **conv_transpose_config
        ))

    def test_conv_transpose_2d_block(self):
        self.assertTrue(butane.nn.ConvTranspose2dBlock(
            input_dims = [1, 28, 28],
            normalization_type = [torch.nn.BatchNorm2d, torch.nn.LayerNorm],
            **conv_transpose_config
        ))

    def test_conv_transpose_3d_block(self):
        self.assertTrue(butane.nn.ConvTranspose3dBlock(
            input_dims = [1, 28, 28, 28],
            normalization_type = [torch.nn.BatchNorm3d, torch.nn.LayerNorm],
            **conv_transpose_config
        ))

    def test_conv_upsample_1d_block(self):
        self.assertTrue(butane.nn.ConvUpsample1dBlock(
            input_dims = [1, 28],
            normalization_type = [torch.nn.BatchNorm1d, torch.nn.LayerNorm],
            **conv_upsample_config
        ))

    def test_conv_upsample_2d_block(self):
        self.assertTrue(butane.nn.ConvUpsample2dBlock(
            input_dims = [1, 28, 28],
            normalization_type = [torch.nn.BatchNorm2d, torch.nn.LayerNorm],
            **conv_upsample_config
        ))

    def test_conv_upsample_3d_block(self):
        self.assertTrue(butane.nn.ConvUpsample3dBlock(
            input_dims = [1, 28, 28, 28],
            normalization_type = [torch.nn.BatchNorm3d, torch.nn.LayerNorm],
            **conv_upsample_config
        ))

if __name__ == '__main__':
    unittest.main()
