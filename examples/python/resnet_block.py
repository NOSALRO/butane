import torch
import torch.nn as nn
import butane
from functools import partial

print(butane.nn.ConvTranspose2dWithRefinmentBlock([1,27,28], [64, 128]))
# block = butane.nn.Residual1dBlock(
#         [64, 32],
#         channels=128,
#         conv_stride=2,
#         pool=torch.nn.MaxPool1d,
#         pool_kernels=3,
#         normalization_type = partial(torch.nn.GroupNorm, num_groups=2),
#         normalization = [True, False]
#     )
# x = torch.randn(8, 64, 32)
# output = block(x)
# print(output.shape)  # Output tensor shape should be [8, 128, 16, 16]

# block = butane.nn.Residual2dBlock(
#         [64, 32, 32],
#         channels=128,
#         activation_function = torch.nn.SiLU(),
#         conv_stride=1,
#         pool=torch.nn.MaxPool2d,
#         pool_kernels=3,
#         normalization_type = partial(torch.nn.GroupNorm, num_groups=32),
#         normalization = [True, False],
#         shortcut_normalization_type = partial(torch.nn.GroupNorm, num_groups=2),
#         shortcut_normalization = True
#     )
# x = torch.randn(8, 64, 32, 32)
# print(block)
# output = block(x)
# print(output.shape)

# block = butane.nn.Residual3dBlock(
#         [64, 32, 32, 32],
#         channels=128,
#         conv_stride=2,
#         pool=torch.nn.MaxPool3d,
#         pool_kernels=3,
#         normalization_type = partial(torch.nn.GroupNorm, num_groups=2),
#         normalization = [True, False],
#     )
# x = torch.randn(8, 64, 32, 32, 32)
# output = block(x)
# print(output.shape)


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, time_channels: int,
#                n_groups: int = 32, dropout: float = 0.1):

#        super().__init__()

#        self.norm1 = nn.GroupNorm(n_groups, in_channels)
#        self.act1 = torch.nn.GELU()
#        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

#        self.norm2 = nn.GroupNorm(n_groups, out_channels)
#        self.act2 = torch.nn.GELU()
#        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))


#        if in_channels != out_channels:
#            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
#        else:
#            self.shortcut = nn.Identity()


#        self.time_emb = nn.Linear(time_channels, out_channels)
#        self.time_act = torch.nn.GELU()

#        self.dropout = nn.Dropout(dropout)


# print(ResidualBlock(64, 128, 256))
