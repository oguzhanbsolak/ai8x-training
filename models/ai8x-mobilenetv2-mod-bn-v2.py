"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
import math

import ai8x
import ai8x_blocks
import math

__all__ = ['mobilenetv2_ai8x']

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    #return nn.Sequential(
    #    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
    #    nn.BatchNorm2d(oup),
    #    nn.ReLU6(inplace=True) # TODO: ReLU6 or ReLU?
    #)
    print("Stride has been overridden to 1")
    seq = nn.Sequential(
        ai8x.FusedConv2dBNReLU(inp, oup, 3, stride=1, padding=1, bias=True),
        ai8x.MaxPool2d(2, 2),
    )
    return seq


def conv_1x1_bn(inp, oup):
    #return nn.Sequential(
    #    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    #    nn.BatchNorm2d(oup),
    #    nn.ReLU6(inplace=True)
    #)
    return ai8x.FusedConv2dBNReLU(inp, oup, 1, stride=1, padding=0, bias=True)

def conv_1x1_no_bn(inp, oup):
    #return nn.Sequential(
    #    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    #    nn.ReLU6(inplace=True)
    #)
    return ai8x.FusedConv2dReLU(inp, oup, 1, stride=1, padding=0, bias=False)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if self.identity:
            self.resid = ai8x.Add()

        if expand_ratio == 1:
            if stride == 2:
                self.conv = nn.Sequential(
                    # dw
                    #nn.MaxPool2d(2, 2, 0),
                    #nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                    #nn.BatchNorm2d(hidden_dim),
                    #nn.ReLU6(inplace=True),# changed from nn.ReLU6(inplace=True),
                    ai8x.FusedMaxPoolDepthwiseConv2dBNReLU(hidden_dim, hidden_dim, 3, pool_size=2, padding=1, bias=True),
                    # pw-linear
                    ai8x.FusedConv2dBN(hidden_dim, oup, 1, stride=1, padding=0, bias=True),

                )
            else:
                self.conv = nn.Sequential(
                    # dw

                    #nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                    #nn.BatchNorm2d(hidden_dim),
                    #nn.ReLU6(inplace=True),# changed from nn.ReLU6(inplace=True),
                    ai8x.FusedDepthwiseConv2dBNReLU(hidden_dim, hidden_dim, 3, padding=1, stride=1, bias=True),
                    # pw-linear
                    ai8x.FusedConv2dBN(hidden_dim, oup, 1, stride=1, padding=0, bias=True),
                    #nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    #nn.BatchNorm2d(oup),
                )
        else:
            if stride == 2:
                self.conv = nn.Sequential(
                    # pw
                    #nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    #nn.BatchNorm2d(hidden_dim),
                    #nn.ReLU6(inplace=True),# changed from nn.ReLU6(inplace=True),
                    ai8x.FusedConv2dBNReLU(inp, hidden_dim, 1, stride=1, padding=0, bias=True),
                    # dw
                    #nn.MaxPool2d(2, 2, 0),
                    #nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                    #nn.BatchNorm2d(hidden_dim),
                    #nn.ReLU6(inplace=True),# changed from nn.ReLU6(inplace=True),
                    ai8x.FusedMaxPoolDepthwiseConv2dBNReLU(hidden_dim, hidden_dim, 3, pool_size=2, padding=1, bias=True),
                    # pw-linear
                    #nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    #nn.BatchNorm2d(oup),
                    ai8x.FusedConv2dBN(hidden_dim, oup, 1, stride=1, padding=0, bias=True),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    #nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    #nn.BatchNorm2d(hidden_dim),
                    #nn.ReLU6(inplace=True),
                    ai8x.FusedConv2dBNReLU(inp, hidden_dim, 1, stride=1, padding=0, bias=True),
                    # dw
                    #nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                    #nn.BatchNorm2d(hidden_dim),
                    #nn.ReLU6(inplace=True),
                    ai8x.FusedDepthwiseConv2dBNReLU(hidden_dim, hidden_dim, 3, padding=1, stride=1, bias=True),
                    # pw-linear
                    #nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    #nn.BatchNorm2d(oup),
                    ai8x.FusedConv2dBN(hidden_dim, oup, 1, stride=1, padding=0, bias=True),
                )

    def forward(self, x):
        if self.identity:
            #print(x.max(), x.min(), "identity")
            #return x + self.conv(x)
            return self.resid(x, self.conv(x))
        else:
            #print(x.max(), x.min(), "no identity")
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=0.5, dimensions=(224,224)):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        print("Width Mult: ", width_mult)
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        #self.input_quantizer_hold = ai8x.Add()
        avg_pool_size = int(dimensions[0]/32)
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        #for m in self.features.modules():
        #    m.register_forward_hook(register_histogram_hook)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        # TODO: For deployment we can't use bn at here as output_channel = 1280
        #self.conv = conv_1x1_no_bn(input_channel, output_channel)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #TODO:I think I can handle this at yaml level
        #self.classifier = nn.Linear(output_channel, num_classes)
        #self.classifier = nn.Conv2d(output_channel, num_classes, 1, 1, 0, bias=False)
        #self.avgpool = ai8x.AvgPool2d(avg_pool_size, avg_pool_size)

        #self.classifier = ai8x.Conv2d(output_channel, num_classes, 1, stride=1, padding=0, bias=True, wide=False)

        self.classifier = ai8x.FusedAvgPoolConv2d(output_channel, num_classes, 1, stride=1, padding=0, bias=True, wide=False, pool_size=avg_pool_size)

        self._initialize_weights()

    def forward(self, x):
        # Does nothing when Floating Point
        #x = self.input_quantizer_hold(x, 0)
        x = self.features(x)
        #print(x.max(), x.min(), "features")
        x = self.conv(x)
        #print(x.max(), x.min(), "conv")
        #x = self.avgpool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def ai8x_mobilenetv2_mod_bn_v2(pretrained=False, dimensions=(112,112), **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(dimensions=dimensions, width_mult=1.0)

def pt_mobilenetv2(pretrained=False, dimensions=(112,112), **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=pretrained)
    print("Model: ", model)
    return model


models = [
    {
        'name': 'ai8x_mobilenetv2_mod_bn_v2',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'pt_mobilenetv2',
        'min_input': 1,
        'dim': 2,
    }
]