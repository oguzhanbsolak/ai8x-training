###################################################################################################
#
# Copyright (C) 2024 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################


from torch import nn

import ai8x


class ResnetBlock(nn.Module):
    def __init__(
            self,
            in_channels=64,
            mid_channels=64,
            expansion=1,
            **kwargs
    ):
        super().__init__()
        self.conv1 = ai8x.FusedConv2dBNReLU(in_channels,
                                            mid_channels, 3,
                                            padding=1, **kwargs)
        self.conv2 = ai8x.FusedConv2dBN(mid_channels,
                                        mid_channels*expansion,
                                        3, padding=1, **kwargs)
        self.combine = ai8x.Add()
        self.activate = nn.ReLU()
        self.adjust = None
        if in_channels != (mid_channels*expansion):
            self.adjust = ai8x.Conv2d(in_channels, mid_channels*expansion,
                                      1, padding=0, **kwargs)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.adjust is not None:
            x = self.adjust(x)
        y = self.combine(y, x)
        return self.activate(y)


class AI87ResnetDataFolding(nn.Module):
    """
    Resnet Model for folded 48x80x60 input
    """
    def __init__(
            self,
            num_classes=2,
            num_channels=48,
            mid_channels=64,
            fc_channels=128,
            dimensions=(80, 60),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()
        # Initial layer input=(80x60) output=(80x60)
        self.layer0 = ai8x.FusedConv2dBNReLU(num_channels, mid_channels, 3, padding=1, **kwargs)
        
        # BLOCK1 input=(80x60) output = (40x30)
        self.maxpool1 = ai8x.MaxPool2d(2, 2)
        self.block1_1 = ResnetBlock(mid_channels, mid_channels, 1, **kwargs)
        self.block1_2 = ResnetBlock(mid_channels, mid_channels, 1, **kwargs)

        # BLOCK2 input=(40x30) output = (20x15)
        self.maxpool2 = ai8x.MaxPool2d(2, 2)
        self.block2_1 = ResnetBlock(mid_channels, mid_channels, 1, **kwargs)
        self.block2_2 = ResnetBlock(mid_channels, mid_channels, 1, **kwargs)

        # BLOCK3 input=(20x15) output = (10x7)
        self.maxpool3 = ai8x.MaxPool2d(2, 2)
        self.block3_1 = ResnetBlock(mid_channels, fc_channels, 1, **kwargs)
        self.block3_2 = ResnetBlock(fc_channels, fc_channels, 1, **kwargs)

        # BLOCK4 input=(10x7) output = (5x3)
        self.maxpool4 = ai8x.MaxPool2d(2, 2)
        self.block4_1 = ResnetBlock(fc_channels, fc_channels, 1, **kwargs)
        self.block4_2 = ResnetBlock(fc_channels, fc_channels, 1, **kwargs)

        # Final Block input=(5x3)
        self.avgpool = ai8x.AvgPool2d(kernel_size=(5,3), stride=(1,1))
        self.fc1 = ai8x.FusedLinearReLU(in_features=fc_channels, out_features=fc_channels, 
                                        bias=bias, **kwargs)
        self.fc2 = ai8x.Linear(in_features=fc_channels, out_features=num_classes, 
                               bias=bias, wide=True, **kwargs)
        
    def forward(self, x):
        # Init Layer
        x = self.layer0(x)
        # BLOCK1
        x = self.maxpool1(x)
        x = self.block1_1(x)
        x = self.block1_2(x)
        # BLOCK2
        x = self.maxpool2(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        # BLOCK3
        x = self.maxpool3(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        # BLOCK4
        x = self.maxpool4(x)
        x = self.block4_1(x)
        x = self.block4_2(x)
        # Final Block
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class AI87ResnetStreaming(nn.Module):
    """
    Resnet Model for streaming 3x320x240 input
    """
    def __init__(
            self,
            num_classes=2,
            num_channels=3,
            mid_channels=64,
            fc_channels=128,
            dimensions=(320, 240),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()
        self.streaming0 = ai8x.FusedConv2dBNReLU(num_channels, mid_channels, 3, padding=1,
                                                 bias=bias, **kwargs)
        self.streaming1 = ai8x.FusedMaxPoolConv2dBNReLU(mid_channels, mid_channels, 3, padding=1,
                                                        bias=bias, **kwargs)
        self.streaming2 = ai8x.FusedMaxPoolConv2dBNReLU(mid_channels, mid_channels, 3, padding=1,
                                                        bias=bias, **kwargs)
        # BLOCK1 input=(80x60) output = (40x30)
        self.maxpool1 = ai8x.MaxPool2d(2, 2)
        self.block1_1 = ResnetBlock(mid_channels, mid_channels, 1, **kwargs)
        self.block1_2 = ResnetBlock(mid_channels, mid_channels, 1, **kwargs)

        # BLOCK2 input=(40x30) output = (20x15)
        self.maxpool2 = ai8x.MaxPool2d(2, 2)
        self.block2_1 = ResnetBlock(mid_channels, mid_channels, 1, **kwargs)
        self.block2_2 = ResnetBlock(mid_channels, mid_channels, 1, **kwargs)

        # BLOCK3 input=(20x15) output = (10x7)
        self.maxpool3 = ai8x.MaxPool2d(2, 2)
        self.block3_1 = ResnetBlock(mid_channels, fc_channels, 1, **kwargs)
        self.block3_2 = ResnetBlock(fc_channels, fc_channels, 1, **kwargs)

        # BLOCK4 input=(10x7) output = (5x3)
        self.maxpool4 = ai8x.MaxPool2d(2, 2)
        self.block4_1 = ResnetBlock(fc_channels, fc_channels, 1, **kwargs)
        self.block4_2 = ResnetBlock(fc_channels, fc_channels, 1, **kwargs)

        # Final Block input=(5x3)
        self.avgpool = ai8x.AvgPool2d(kernel_size=(5,3), stride=(1,1))
        self.fc = ai8x.Linear(in_features=fc_channels, out_features=num_classes, 
                              bias=bias, wide=True, **kwargs)
        
    def forward(self, x):
        # Streaming Layers
        x = self.streaming0(x)
        x = self.streaming1(x)
        x = self.streaming2(x)
        # BLOCK1
        x = self.maxpool1(x)
        x = self.block1_1(x)
        x = self.block1_2(x)
        # BLOCK2
        x = self.maxpool2(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        # BLOCK3
        x = self.maxpool3(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        # BLOCK4
        x = self.maxpool4(x)
        x = self.block4_1(x)
        x = self.block4_2(x)
        # Final Block
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai87resnetstreaming(pretrained=False, **kwargs):
    """
    Constructs a AI85NetExtraSmall model.
    """
    assert not pretrained
    return AI87ResnetStreaming(**kwargs)

def ai87resnetfolded(pretrained=False, **kwargs):
    """
    Constructs a AI85NetExtraSmall model.
    """
    assert not pretrained
    return AI87ResnetDataFolding(**kwargs)

models = [
    {
        'name': 'ai87resnetstreaming',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai87resnetfolded',
        'min_input': 1,
        'dim': 2,
    },
]
