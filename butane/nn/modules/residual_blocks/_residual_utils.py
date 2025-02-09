from ..conv_blocks import Conv1dBlock, Conv2dBlock, Conv3dBlock


def define_Nd_residual(conv_type: str):
    def inner(cls):
        if conv_type == '1d':
            cls.conv = Conv1dBlock
        elif conv_type == '2d':
            cls.conv = Conv2dBlock
        elif conv_type == '3d':
            cls.conv = Conv3dBlock
        return cls
    return inner
