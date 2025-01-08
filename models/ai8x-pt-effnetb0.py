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
import copy
import ai8x
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from functools import partial
from torchvision.utils import _make_ntuple
from torchvision.models._utils import _make_divisible
from torchvision.ops import StochasticDepth
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models import efficientnet_b0


class Conv2dNormActivation(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]] = 3,
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Optional[Union[int, Tuple[int, ...], str]] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: Union[int, Tuple[int, ...]] = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None,
            conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
        ) -> None:

            if padding is None:
                if isinstance(kernel_size, int) and isinstance(dilation, int):
                    padding = (kernel_size - 1) // 2 * dilation
                else:
                    _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                    kernel_size = _make_ntuple(kernel_size, _conv_dim)
                    dilation = _make_ntuple(dilation, _conv_dim)
                    padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
            if bias is None:
                bias = norm_layer is None

            layers = [
                conv_layer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )
            ]

            if norm_layer is not None:
                layers.append(norm_layer(out_channels))

            if activation_layer is not None:
                params = {} if inplace is None else {"inplace": inplace}
                layers.append(activation_layer(**params))
            super().__init__(*layers)

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)



class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        #activation_layer = nn.SiLU
        activation_layer = nn.ReLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        #layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.ReLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        """
        EfficientNet V1 main class

        Args:
            inverted_residual_setting (Sequence[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                #3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU,
                #activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[MBConvConfig], Optional[int]]:
    inverted_residual_setting: Sequence[MBConvConfig]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(MBConvConfig, width_mult=kwargs.pop("width_mult"), depth_mult=kwargs.pop("depth_mult"))
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def ai8x_pt_effnetb0(pretrained=True, dimensions=(112,112), **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    pool_size = dimensions[0]//32
    #print("Model: ", model)


    # Model from torch
  
    pt_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    state_dict = pt_model.state_dict()
    #print("State Dict: ", state_dict.keys())
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    
    model = EfficientNet(inverted_residual_setting, dropout=0.2, last_channel=last_channel)

    #Model from AI8X
    ai8x_state_dict = model.state_dict()
    #print("AI8X State Dict: ", ai8x_state_dict.keys())
    
    already_loaded = []
    # Load the state dict from the torch model
    for key in state_dict:
        found = False
        for ai8x_key in ai8x_state_dict:
            if ai8x_key in already_loaded or ai8x_key.split(".")[-1] != key.split(".")[-1]:
                continue
            if state_dict[key].shape == ai8x_state_dict[ai8x_key].shape:
                print("Loading: ", key, ai8x_key)
                #if "num_batches_tracked" in key:
                    #print(state_dict[key])
                    #print(ai8x_state_dict[ai8x_key])
                ai8x_state_dict[ai8x_key] = state_dict[key]
                already_loaded.append(ai8x_key)
                found = True
                break
        if not found:
            print("Not Found: ", key)
    model.load_state_dict(ai8x_state_dict)
    return model







models = [
    {
        'name': 'ai8x_pt_effnetb0',
        'min_input': 1,
        'dim': 2,
    },

]