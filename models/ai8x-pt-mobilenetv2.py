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
    seq = nn.Sequential(
        ai8x.FusedConv2dBNReLU6(inp, oup, 3, stride=stride, padding=1, bias=True),
    )
    return seq


def conv_1x1_bn(inp, oup):
    #return nn.Sequential(
    #    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    #    nn.BatchNorm2d(oup),
    #    nn.ReLU6(inplace=True)
    #)
    return ai8x.FusedConv2dBNReLU6(inp, oup, 1, stride=1, padding=0, bias=True)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if self.identity:
            self.resid = ai8x.Add()

        if expand_ratio == 1:

            self.conv = nn.Sequential(
                # dw
                ai8x.FusedDepthwiseConv2dBNReLU6(hidden_dim, hidden_dim, 3, padding=1, stride=stride, bias=True),
                # pw-linear
                ai8x.FusedConv2dBN(hidden_dim, oup, 1, stride=1, padding=0, bias=True),

            )
        else:
            self.conv = nn.Sequential(
                # pw
                ai8x.FusedConv2dBNReLU6(inp, hidden_dim, 1, stride=1, padding=0, bias=True),
                # dw
                ai8x.FusedDepthwiseConv2dBNReLU6(hidden_dim, hidden_dim, 3, padding=1, stride=stride, bias=True),
                # pw-linear
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
        self.input_quantizer_hold = ai8x.ActivationQHolder()
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

        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)

        self.avgpool = ai8x.AvgPool2d(avg_pool_size, avg_pool_size)


        self.classifier = ai8x.Linear(output_channel, num_classes, bias=True)
        self._initialize_weights()

    def forward(self, x):
        x = self.input_quantizer_hold(x, self.input_quantizer_hold.activation_threshold)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)


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

def ai8x_pt_mobilenetv2(pretrained=True, dimensions=(112,112), **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model = MobileNetV2(dimensions=dimensions, width_mult=1.0)


    # Model from torch
    #pt_model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    #state_dict = pt_model.state_dict()

    # Model from AI8X
    #ai8x_state_dict = model.state_dict()
    #print("AI8X State Dict: ", ai8x_state_dict.keys())
    #exit()
    #already_loaded = []
    # Load the state dict from the torch model
    #for key in state_dict:
    #    found = False
    #    for ai8x_key in ai8x_state_dict:
    #        if ai8x_key in already_loaded or ai8x_key.split(".")[-1] != key.split(".")[-1] or (ai8x_key.split(".")[-1] =="bias" and ai8x_key.split(".")[-2] != "bn"):
    #            continue
    #        if state_dict[key].shape == ai8x_state_dict[ai8x_key].shape:
    #            print("Loading: ", key, ai8x_key)
    #            if "num_batches_tracked" in key:
    #                print(state_dict[key])
    #                print(ai8x_state_dict[ai8x_key])
    #            ai8x_state_dict[ai8x_key] = state_dict[key]
    #            already_loaded.append(ai8x_key)
    #            found = True
    #            break
    #    if not found:
    #        print("Not Found: ", key)
   # model.load_state_dict(ai8x_state_dict)


    return model



models = [
    {
        'name': 'ai8x_pt_mobilenetv2',
        'min_input': 1,
        'dim': 2,
    },

]