###################################################################################################
#
# Copyright (C) 2020-2024 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
# pyright: reportOptionalMemberAccess=false, reportPrivateImportUsage=false
# pyright: reportOptionalCall=false, reportOptionalOperand=false
"""
Contains the limits of the MAX78000/MAX78002 implementations and custom PyTorch modules that take
the limits into account.
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.fx import symbolic_trace
import math
from tqdm import tqdm

import devices

dev = None


class normalize:
    """
    Normalize input to either [-128/128, +127/128] or [-128, +127]
    """
    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        if self.args.act_mode_8bit:
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127).div(128.)


class fold:
    """
    Fold data to increase the number of channels. An interlaced approach used in this folding
    as explained in [1].

    [1] https://arxiv.org/pdf/2203.16528.pdf
    """
    def __init__(self, fold_ratio):
        self.fold_ratio = fold_ratio

    def __call__(self, img):
        if self.fold_ratio == 1:
            return img

        img_folded = None
        for i in range(self.fold_ratio):
            for j in range(self.fold_ratio):
                img_subsample = img[:, i::self.fold_ratio, j::self.fold_ratio]
                if img_folded is not None:
                    img_folded = torch.cat((img_folded, img_subsample), dim=0)
                else:
                    img_folded = img_subsample

        return img_folded


def unfold_batch(img_batch, fold_ratio):
    """
    Unfold data to reduce the number of channels. An interlaced approach used in this folding
    as explained in [1]. This operation is the reverse of the transformation implemented
    at ai8x.fold class.

    [1] https://arxiv.org/pdf/2203.16528.pdf
    """
    if fold_ratio == 1:
        return img_batch

    num_out_channels = img_batch.shape[1] // (fold_ratio*fold_ratio)

    img_batch_uf = torch.zeros((img_batch.shape[0], num_out_channels,
                                img_batch.shape[2]*fold_ratio, img_batch.shape[3]*fold_ratio),
                               dtype=img_batch.dtype, device=img_batch.device, requires_grad=False)

    for i in range(fold_ratio):
        for j in range(fold_ratio):
            ch_index_start = num_out_channels*(i*fold_ratio + j)
            ch_index_end = num_out_channels * (i*fold_ratio + j + 1)
            img_batch_uf[:, :, i::fold_ratio, j::fold_ratio] = \
                img_batch[:, ch_index_start:ch_index_end, :, :]

    return img_batch_uf


class QuantizationFunction(Function):
    """
    Custom autograd function
    The forward pass divides by 2**(bits-1) (typically, 128) and rounds the result to the
    nearest integer.
    The backward pass is straight through.
    """
    # pylint: disable=abstract-method

    @staticmethod
    def forward(_, x, bits=8, extra_bit_shift=0, per_channel=False):  # pylint: disable=arguments-differ
        """Forward prop"""
        if dev.fakeactquant:
            if per_channel:
                step_size = (2 ** extra_bit_shift) / (2 ** (bits - 1))
                if len(x.shape) == 2:
                    step_size = step_size.to(x.device).view(1, -1)
                else:
                    step_size = step_size.to(x.device).view(1, -1, 1, 1)  # Reshape for broadcasting
                #print("Step Size.shape", step_size.shape)
                #print("X.shape", x.shape)
                return x.div(step_size).round()
            step_size = 2**(extra_bit_shift) / 2**(bits-1)
            return x.div(step_size).round()
        elif dev.simulate:
            if bits > 1:
                return x.div(2**(bits+extra_bit_shift-1)).add(.5).floor()
            if bits < 1:
                return x.mul(2**(1-bits-extra_bit_shift)).add(.5).floor()
            return x.add(.5).floor()

        factor1 = 2**(bits-extra_bit_shift-1)
        factor2 = 2**(bits-1)
        return x.mul(factor1).add(.5).floor().div(factor2)

    @staticmethod
    def backward(_, x):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x, None, None, None


class QuantizationFunctionwoRound(nn.Module):
    """
    Custom autograd function
    The forward pass divides by 2**(bits-1) (typically, 128) and rounds the result to the
    nearest integer.
    The backward pass is straight through.
    """
    # pylint: disable=abstract-method

    def __init__(self):
        super().__init__()

    def forward(self, x, bits=8, extra_bit_shift=0, per_channel=False):  # pylint: disable=arguments-differ
        """Forward prop"""
        if dev.fakeactquant:
            if per_channel:
                step_size = (2 ** extra_bit_shift) / (2 ** (bits - 1))
                if len(x.shape) == 2:
                    step_size = step_size.to(x.device).view(1, -1)
                else:
                    step_size = step_size.to(x.device).view(1, -1, 1, 1)  # Reshape for broadcasting
                #print("Step Size.shape", step_size.shape)
                #print("X.shape", x.shape)
                return x.div(step_size)
            step_size = 2**(extra_bit_shift) / 2**(bits-1)
            return x.div(step_size)
        else:
            print("Not implemented")
            exit(1)


class DequantizationFunctionwoRound(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, bits, scale_factor, per_channel=False):
        """
        Dequantize the tensor with the given scale factor
        """
        if per_channel:
            step_size = (2.0 ** scale_factor) / (2.0 ** (bits - 1))
            if len(x.shape) == 2:
                step_size = step_size.to(x.device).view(1, -1)
            else:
                step_size = step_size.to(x.device).view(1, -1, 1, 1)
            return x.mul(step_size)
        step_size = (2.**scale_factor) / 2.**(bits-1)
        return x.mul(step_size)


class RoundClamp(Function):
    """
    Custom autograd function
    The forward pass divides by 2**(bits-1) (typically, 128) and rounds the result to the
    nearest integer.
    The backward pass is straight through.
    """
    # pylint: disable=abstract-method

    @staticmethod
    def forward(ctx, x, bits):  # pylint: disable=arguments-differ
        """Forward prop"""
        x_out = x.round().clamp(min=-2**(bits-1), max=2**(bits-1) - 1)
        ctx.save_for_backward(torch.clamp(x - x_out, min=-1, max=1))
        ctx.bits = bits

        return x_out

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        diff = ctx.saved_tensors[0]
        scale = 1 + torch.sign(grad_output)*diff*0.01
        return grad_output * scale, None


class RoundClampSTE(Function):
    """
    Custom autograd function
    The forward pass divides by 2**(bits-1) (typically, 128) and rounds the result to the
    nearest integer.
    The backward pass is straight through.
    """
    # pylint: disable=abstract-method

    @staticmethod
    def forward(ctx, x, bits):  # pylint: disable=arguments-differ
        """Forward prop"""
        x_out = x.round().clamp(min=-2**(bits-1), max=2**(bits-1) - 1)
        #ctx.save_for_backward(torch.clamp(x - x_out, min=-1, max=1))
        #ctx.bits = bits

        return x_out

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        #diff = ctx.saved_tensors[0]
        #scale = 1 + torch.sign(grad_output)*diff*0.01
        #return grad_output * scale, None
        return grad_output, None


class FakeQuantizeWeightv2(nn.Module):
    """
    Fake quantization module
    """
    def __init__(self, bits=8, per_channel=False):
        super().__init__()
        self.bits = bits
        self.per_channel = per_channel
        self.q_func = QuantizationFunctionwoRound()
        self.dq_func = DequantizationFunctionwoRound()
        #print("Clamp", clamp)

    def forward(self, x, scale):  # pylint: disable=arguments-differ
        """Forward prop"""
        #print("Scale", scale)
        #print("Before Quantization")
        #for i in range(x.shape[1]):
        #    print("Channel", i, x[:, i].min(), x[:, i].max())
        x = self.q_func(x, self.bits, scale, self.per_channel)
        #print("After Quantization")
        #for i in range(x.shape[1]):
        #    print("Channel", i, x[:, i].min(), x[:, i].max())
        #print(x.min(), x.max())
        #x = RoundClamp.apply(x, self.bits)
        x = RoundClampSTE.apply(x, self.bits)
        #print(x.min(), x.max())
        #print("After Clamp")
        #for i in range(x.shape[1]):
        #    print("Channel", i, x[:, i].min(), x[:, i].max())
        x = self.dq_func(x, self.bits, scale, self.per_channel)
        #for i in range(x.shape[1]):
        #    print("Channel", i, x[:, i].min(), x[:, i].max())
        return x


class FakeQuantizeActv2(nn.Module):
    """
    Fake quantization module
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.q_func = QuantizationFunctionwoRound()
        self.dq_func = DequantizationFunctionwoRound()

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        #print("ACT", self.scale)
        #print("Before Quantization", x.min(), x.max())
        #x = ai8x.QuantizationFunctionwoRound(x, ai8x.dev.ACTIVATION_BITS, self.scale)
        x = self.q_func(x, dev.ACTIVATION_BITS, self.scale)
        #print("After Quantization", x.min(), x.max())
        x = RoundClamp.apply(x, dev.ACTIVATION_BITS)
        #print("After Clamp", x.min(), x.max())
        #x = ai8x.DequantizationFunctionwoRound(x, ai8x.dev.ACTIVATION_BITS, self.scale)
        x = self.dq_func(x, dev.ACTIVATION_BITS, self.scale)
        #print("After Dequantization", x.min(), x.max())
        return x


class FakeQuantizePoolv2(nn.Module):
    """
    Fake quantization module
    """
    def __init__(self):
        super().__init__()
        self.q_func = QuantizationFunctionwoRound()
        self.dq_func = DequantizationFunctionwoRound()

    def forward(self, x, scale):  # pylint: disable=arguments-differ
        """Forward prop"""
        #print("ACT", self.scale)
        #print("Before Quantization", x.min(), x.max())
        #x = ai8x.QuantizationFunctionwoRound(x, ai8x.dev.ACTIVATION_BITS, self.scale)
        x = self.q_func(x, dev.ACTIVATION_BITS, scale)
        #print("After Quantization", x.min(), x.max())
        x = RoundClamp.apply(x, dev.ACTIVATION_BITS)
        #print("After Clamp", x.min(), x.max())
        #x = ai8x.DequantizationFunctionwoRound(x, ai8x.dev.ACTIVATION_BITS, self.scale)
        x = self.dq_func(x, dev.ACTIVATION_BITS, scale)
        #print("After Dequantization", x.min(), x.max())
        return x


class Quantize(nn.Module):
    """
    Post-activation integer quantization module
    Apply the custom autograd function
    """
    def __init__(self, num_bits=8, num_extra_bit_shift=0):
        super().__init__()
        self.num_bits = num_bits
        self.num_extra_bit_shift = num_extra_bit_shift

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return QuantizationFunction.apply(x, self.num_bits, self.num_extra_bit_shift)


class FakeQuantizeWeight(nn.Module):
    """
    Fake quantization module
    """
    def __init__(self, clamp=None, bits=8, per_channel=False):
        super().__init__()
        self.clamp = clamp
        self.bits = bits
        self.per_channel = per_channel
        #print("Clamp", clamp)

    def forward(self, x, scale):  # pylint: disable=arguments-differ
        """Forward prop"""
        #print("Scale", scale)
        #print("Before Quantization")
        #for i in range(x.shape[1]):
        #    print("Channel", i, x[:, i].min(), x[:, i].max())
        x = QuantizationFunction.apply(x, self.bits, scale, self.per_channel)
        #print("After Quantization")
        #for i in range(x.shape[1]):
        #    print("Channel", i, x[:, i].min(), x[:, i].max())
        x = self.clamp(x)
        #print("After Clamp")
        #for i in range(x.shape[1]):
        #    print("Channel", i, x[:, i].min(), x[:, i].max())
        x = DequantizationFunction.apply(x, self.bits, scale, self.per_channel)
        #for i in range(x.shape[1]):
        #    print("Channel", i, x[:, i].min(), x[:, i].max())
        return x


class FakeQuantizeAct(nn.Module):
    """
    Fake quantization module
    """
    def __init__(self, scale, clamp=None):
        super().__init__()
        self.scale = scale
        self.clamp = clamp

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        #print("ACT", self.scale)
        #print("Before Quantization", x.min(), x.max())
        x = QuantizationFunction.apply(x, dev.ACTIVATION_BITS, self.scale)
        #print("After Quantization", x.min(), x.max())
        x = self.clamp(x)
        #print("After Clamp", x.min(), x.max())
        x = DequantizationFunction.apply(x, dev.ACTIVATION_BITS, self.scale)
        #print("After Dequantization", x.min(), x.max())
        return x



class FakeQuantizePool(nn.Module):
    """
    Fake quantization module
    """
    def __init__(self, clamp=None):
        super().__init__()
        self.clamp = clamp

    def forward(self, x, scale):  # pylint: disable=arguments-differ
        """Forward prop"""
        #print("POOL", scale)
        #print("Before Quantization", x.min(), x.max())
        x = QuantizationFunction.apply(x, dev.ACTIVATION_BITS, scale)
        #print("After Quantization", x.min(), x.max())
        x = self.clamp(x)
        #print("After Clamp", x.min(), x.max())
        x = DequantizationFunction.apply(x, dev.ACTIVATION_BITS, scale)
        #print("After Dequantization", x.min(), x.max())
        return x


class FakeGradModifierAct(Function):

    @staticmethod
    def forward(ctx, x, function):
        x_out = function(x)
        ctx.save_for_backward(x - x_out)
        ctx.bits = torch.log2(torch.tensor(dev.ACTIVATION_BITS / 8))
        ctx.scale = function.scale
        #ctx.sigma = torch.clamp(1/(2*torch.std(x)), max=0.5)  # std(x) = 1 => sigma = 0.5, std(x) = 2 => sigma = 0.25
        #print("sigma act", torch.std(x))

        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        diff = ctx.saved_tensors[0]
        scale = 1 + torch.sign(grad_output)*diff* torch.exp2(ctx.bits - ctx.scale)

        return grad_output * scale, None, None

class FakeGradModifierPool(Function):

        @staticmethod
        def forward(ctx, x, scale, function):
            x_out = function(x, scale)
            ctx.save_for_backward(x - x_out)
            ctx.bits = torch.log2(torch.tensor(dev.ACTIVATION_BITS / 8))
            ctx.scale = scale
            #ctx.sigma = torch.clamp(1/(2*torch.std(x)), max=0.5)  # std(x) = 1 => sigma = 0.5, std(x) = 2 => sigma = 0.25
            #print("sigma pool", torch.std(x))

            return x_out

        @staticmethod
        def backward(ctx, grad_output):
            diff = ctx.saved_tensors[0]
            scale = 1 + torch.sign(grad_output)*diff* torch.exp2(ctx.bits - ctx.scale)

            return grad_output * scale, None, None, None


class FakeGradModifierWeight(Function):
    @staticmethod
    def forward(ctx, x, scale, function):
        x_out = function(x, scale)
        ctx.save_for_backward(x - x_out)
        ctx.bits = torch.log2(torch.tensor(function.bits / 8))
        ctx.scale = scale
        #ctx.sigma = torch.clamp(1/(2*torch.std(x)), max=0.5)  # std(x) = 1 => sigma = 0.5, std(x) = 2 => sigma = 0.25
        #print("sigma weight", torch.std(x))

        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        diff = ctx.saved_tensors[0]
        scale = 1 + torch.sign(grad_output)*diff* torch.exp2(ctx.bits - ctx.scale)

        return grad_output * scale, None, None, None

class FloorFunction(Function):
    """
    Custom MAX78000/MAX78002 autograd function
    The forward pass returns the integer floor.
    The backward pass is straight through.
    """
    # pylint: disable=abstract-method

    @staticmethod
    def forward(_, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x.floor()

    @staticmethod
    def backward(_, x):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x


class AvgPoolFloorFunction(Function):
    """
    Custom MAX78000/MAX78002 autograd function
    The forward pass returns the integer floor for positive numbers and integer
    ceil for negative numbers.
    The backward pass is straight through.
    """
    # pylint: disable=abstract-method

    @staticmethod
    def forward(_, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return torch.where(x > 0, torch.floor(x), torch.ceil(x))

    @staticmethod
    def backward(_, x):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x


class Floor(nn.Module):
    """
    Post-pooling integer quantization module
    Apply the custom autograd function
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return FloorFunction.apply(x)


class AvgPoolFloor(nn.Module):
    """
    Post-pooling integer quantization module
    Apply the custom autograd function
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return AvgPoolFloorFunction.apply(x)


class FloorONNX(nn.Module):
    """
    Post-pooling integer quantization module
    Apply the custom autograd function
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x.floor()


class RoundFunction(Function):
    """
    Custom MAX78000/MAX78002 autograd function
    The forward pass returns the integer rounded.
    The backward pass is straight through.
    """
    # pylint: disable=abstract-method

    @staticmethod
    def forward(_, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x.round()

    @staticmethod
    def backward(_, x):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x


class Round(nn.Module):
    """
    Post-pooling integer quantization module
    Apply the custom autograd function
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return RoundFunction.apply(x)


class Clamp(nn.Module):
    """
    Post-Activation Clamping Module
    Clamp the output to the given range (typically, [-128, +127])
    """
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = x.clamp(min=self.min_val)
        return x.clamp(max=self.max_val)




class Scaler(nn.Module):
    """
    Scaler module that considers integer quantization
    Apply the custom autograd function
    """
    def forward(self, x, s):  # pylint: disable=arguments-differ
        """Forward prop"""
        if dev.simulate:
            return FloorFunction.apply(x.mul(s))
        return x.mul(s)


class ScalerONNX(nn.Module):
    """
    Scaler module that considers integer quantization
    Apply the custom autograd function
    """
    def forward(self, x, s):  # pylint: disable=arguments-differ
        """Forward prop"""
        if dev.simulate:
            return x.mul(s).floor()
        return x.mul(s)


class ID3(nn.Module):
    """
    ID forward function with 3 arguments
    """
    def forward(self, x, _):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x


class RoundQat(nn.Module):
    """
    Round function for AvgPool in QAT mode
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        factor = 2**(dev.ACTIVATION_BITS - 1)
        return RoundFunction.apply(x.mul(factor)).div(factor)


class RoundQatONNX(nn.Module):
    """
    Round function for AvgPool in QAT mode
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        factor = 2**(dev.ACTIVATION_BITS - 1)
        return x.mul(factor).round().div(factor)


class FloorQat(nn.Module):
    """
    Floor function for AvgPool in QAT mode
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        factor = 2**(dev.ACTIVATION_BITS - 1)
        return AvgPoolFloorFunction.apply(x.mul(factor)).div(factor)


class FloorQatONNX(nn.Module):
    """
    Floor function for AvgPool in QAT mode
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        factor = 2**(dev.ACTIVATION_BITS - 1)
        return x.mul(factor).floor().div(factor)


def quantize_clamp(wide, quantize_activation=False, clamp_activation=False, weight_bits=8):
    """
    Return new Quantization and Clamp objects.
    """
    if dev.simulate or dev.fakeactquant:
        if not wide:
            quantize = Quantize(num_bits=dev.DATA_BITS)
            clamp = Clamp(
                min_val=-(2**(dev.ACTIVATION_BITS-1)),
                max_val=2**(dev.ACTIVATION_BITS-1)-1,
            )
        else:
            quantize = Quantize(num_bits=dev.DATA_BITS - weight_bits + 1)
            clamp = Clamp(
                min_val=-(2**(dev.FULL_ACC_BITS-1)),
                max_val=2**(dev.FULL_ACC_BITS-1)-1,
            )
    else:
        if quantize_activation:
            if not wide:
                quantize = Quantize(num_bits=dev.ACTIVATION_BITS)
            else:
                quantize = Quantize(num_bits=dev.WIDE_LAYER_RESOLUTION_BITS)
        else:
            quantize = Empty()

        if clamp_activation:
            if not wide:
                clamp = Clamp(  # Do not combine with ReLU
                    min_val=-1.,
                    max_val=(2.**(dev.ACTIVATION_BITS-1)-1)/(2.**(dev.ACTIVATION_BITS-1)),
                )
            else:
                clamp = Clamp(
                    min_val=-(2.**((dev.FULL_ACC_BITS-2*(dev.DATA_BITS-1))-1)),
                    max_val=2.**((dev.FULL_ACC_BITS-2*(dev.DATA_BITS-1))-1),
                )
        else:
            clamp = Empty()

    return quantize, clamp


def quantize_clamp_pool(pooling, quantize_activation=False, clamp_activation=False):
    """
    Return new Quantization and Clamp objects for pooling.
    """
    if dev.simulate or dev.fakeactquant:
        if pooling == 'Avg':
            quantize = Round() if dev.round_avg else AvgPoolFloor()
            clamp = Clamp(
                min_val=-(2**(dev.DATA_BITS-1)),
                max_val=2**(dev.DATA_BITS-1)-1,
            )
        else:  # Max, None
            quantize = Empty()
            clamp = Empty()
    else:
        quantize = Empty()
        if pooling == 'Avg':
            if quantize_activation:
                quantize = RoundQat() if dev.round_avg else FloorQat()
            if clamp_activation:
                clamp = Clamp(min_val=-1., max_val=127./128.)
            else:
                clamp = Empty()
        else:  # Max, None
            clamp = Empty()

    return quantize, clamp


def find_pool_scale(x):
    """
    Estimate the scale factor from the pre-pooling tensor
    """
    scale_factor = x.abs().max()
    scale_factor = scale_factor.log2().ceil().clamp(min=0.)
    return scale_factor

class DequantizationFunction(Function):

    @staticmethod
    def forward(_, x, bits, scale_factor, per_channel=False):
        """
        Dequantize the tensor with the given scale factor
        """
        if per_channel:
            step_size = (2.0 ** scale_factor) / (2.0 ** (bits - 1))
            if len(x.shape) == 2:
                step_size = step_size.to(x.device).view(1, -1)
            else:
                step_size = step_size.to(x.device).view(1, -1, 1, 1)
            return x.mul(step_size)
        step_size = (2.**scale_factor) / 2.**(bits-1)
        return x.mul(step_size)

    def backward(_, x):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x, None, None, None

###From TQT implementation
def number_to_tensor(x, t):
    r'''
    Turn x in to a tensor with data type like tensor t.
    '''
    return torch.tensor(x).type_as(t)


class RoundToEven(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.ones(grad_output.shape).type_as(grad_output)


class Ceil(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.ceil(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return torch.ones(grad_output.shape).type_as(grad_output)


class qSigned(Function):
    @staticmethod
    def forward(ctx, x, log2_t, bit_width, mode=False):
        r'''
        See TQT paper's Eqn. (4)
        '''
        bit_max = 2.**(bit_width - 1)
        n = -bit_max
        p = bit_max - 1
        s = 2.**Ceil.apply(log2_t) / bit_max
        if not mode:
            q = torch.clamp(RoundToEven.apply(x / s), n, p) * s
        else:
            s2 = 1 / bit_max
            
            q = torch.clamp(RoundToEven.apply(x / s), n, p) * s2
            #TODO: Check if this is correct = Seems Correct
        ctx.save_for_backward(x / s, s, number_to_tensor(n, x),
                              number_to_tensor(p, x))
        return q

    @staticmethod
    def backward(ctx, grad_output):
        x_div_s, s, n, p = ctx.saved_tensors
        rounded = RoundToEven.apply(x_div_s)
        cmp0 = (n <= rounded) & (rounded <= p)
        cmp1 = rounded < n
        cmp2 = rounded > p
        grad_s = (rounded - x_div_s) * cmp0 + n * cmp1 + p * cmp2
        grad_log_2_t = math.log(2) * grad_s * s
        grad_x = cmp0 * 1.0
        return grad_output * grad_x, grad_output * grad_log_2_t, None, None

class PoTAlpha(Function):

    @staticmethod
    def forward(_, alpha):
        """
        Dequantize the tensor with the given scale factor
        """
        return alpha.ceil().clamp(min=-15., max=15.)

    def backward(_, x):  # pylint: disable=arguments-differ
        """Backprop"""
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return torch.ones(x.shape).type_as(x)


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def quantize_clamp_parameters(weight_bits, bias_bits):
    """
    Return new Quantization and Clamp objects for weight and bias parameters
    """
    if dev.simulate:
        quantize_weight = Quantize(num_bits=weight_bits-dev.DATA_BITS+1)
        quantize_bias = Quantize(num_bits=2*(weight_bits-dev.DATA_BITS)+1)
        clamp_weight = Empty()
        clamp_bias = Empty()
    elif dev.fakeactquant:
        quantize_weight = Quantize(num_bits=weight_bits)
        quantize_bias = Quantize(num_bits=bias_bits)
        #TODO: Check if this is correct
        clamp_weight = Clamp(
            min_val=-(2.**(weight_bits-1)),
            max_val=2.**(weight_bits-1)-1,
        )
        clamp_bias = Clamp(
            min_val=-(2.**(bias_bits-1)),
            max_val=2.**(bias_bits-1)-1,
        )
    else:
        if weight_bits == 0 and bias_bits == 0:
            quantize_weight = Empty()
            quantize_bias = Empty()
            clamp_weight = Empty()
            clamp_bias = Empty()
        else:
            quantize_weight = Quantize(num_bits=weight_bits)
            quantize_bias = Quantize(num_bits=bias_bits)
            clamp_weight = Clamp(min_val=-1.,
                                 max_val=(2.**(weight_bits-1)-1)/(2.**(weight_bits-1)))
            clamp_bias = Clamp(min_val=-1., max_val=(2.**(bias_bits-1)-1)/(2.**(bias_bits-1)))

    return quantize_weight, quantize_bias, clamp_weight, clamp_bias


class OutputShiftPassthrough(nn.Module):
    """
    Return output_shift when not using quantization-aware training.
    """
    def forward(self, _, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x


class OutputShiftLimit(nn.Module):
    """
    Calculate the clamped output shift when adjusting during quantization-aware training.
    """
    def __init__(self, shift_quantile=1.0, per_channel=False):
        super().__init__()
        self.shift_quantile = shift_quantile
        self.per_channel = per_channel
        #print("Per Channel", per_channel)

    def forward(self, x, _):  # pylint: disable=arguments-differ
        """Forward prop"""
        if self.per_channel:
            # Get the per-channel limit
            if len(x.shape) == 2:
                abs_x = x.abs().permute(1, 0).reshape(x.shape[1], -1)
            else:
                abs_x = x.abs().permute(1, 0, 2, 3).reshape(x.shape[1], -1)

            # Compute quantiles along the batch dimension (dim=0)
            limits = torch.quantile(abs_x, self.shift_quantile, dim=1, keepdim=True)  # Assuming quantile supports `dim`
            # Compute -(1. / limit).log2().floor().clamp(min=-15, max=15)
            return -(1. / limits).log2().floor().clamp(min=-15., max=15.)


        limit = torch.quantile(x.abs(), self.shift_quantile)
        return -(1./limit).log2().floor().clamp(min=-15., max=15.)


class OutputShiftONNX(nn.Module):
    """
    Calculate the clamped output shift when adjusting during quantization-aware training.
    """
    def forward(self, x, _):  # pylint: disable=arguments-differ
        """Forward prop"""
        return -(1./x.abs().max()).log2().floor().clamp(min=-15., max=15.)


class One(nn.Module):
    """
    Return 1.
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return torch.ones(1).to(x.device)


class WeightScale(nn.Module):
    """
    Calculate the weight scale (reciprocal of 2 to the power of the output shift)
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return torch.exp2(-x)


class WeightScaleONNX(nn.Module):
    """
    Calculate the weight scale (reciprocal of 2 to the power of the output shift)
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return 2.**(-x)


class OutputScale(nn.Module):
    """
    Calculate the output scale (2 to the power of the output shift)
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return torch.exp2(x)


class OutputScaleONNX(nn.Module):
    """
    Calculate the output scale (2 to the power of the output shift)
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return 2.**x


class Abs(nn.Module):
    """
    Return abs(x)
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return torch.abs_(x)  # abs_() is the in-place version


class Empty(nn.Module):
    """
    Do nothing
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x


def get_activation(activation=None):
    """
    Return the selected `activation` class ('ReLU', 'Abs', None)
    """
    if activation == 'ReLU':
        return nn.ReLU(inplace=True)
    if activation == 'Abs':
        assert dev.device != 84
        return Abs()
    if activation == 'ReLU6':
        return nn.ReLU6(inplace=True)
    return Empty()


def histogram(inp, bins):
    """
    CUDA compatible histogram calculation
    """
    minimum, maximum = inp.min(), inp.max()
    counts = torch.histc(inp, bins, min=minimum, max=maximum).cpu()
    boundaries = torch.linspace(minimum, maximum, bins + 1)
    return counts, boundaries


def calc_q_error(module, threshold, bits, eps=1e-9):
    """
    Activation quantization error calculation
    """
    quantized_hist = module.hist[1].clone()
    quantized_hist = torch.round((quantized_hist / (threshold + eps)) * 2**(bits-1))
    quantized_hist = torch.clamp(quantized_hist, -2**(bits-1), 2**(bits-1)-1)
    quantized_hist = (quantized_hist * (threshold + eps) / 2**(bits-1))
    err = torch.sum(((quantized_hist - module.hist[1])**2)*module.hist[0]) \
        / torch.sum(module.hist[0])

    return err


def calc_q_error_weight(weight, threshold, bits, qfunction):
    """
    Weight quantization error calculation
    """
    logt = threshold.log2()
    quantized_hist = qfunction.apply(weight, logt, bits)
    err = torch.sum(((quantized_hist - weight)**2)) / torch.numel(weight)

    return err



def _merge_hist(module):
    """
    Merge histograms of activations
    """
    bins_to_stack = []
    for hist in module.hist:
        bins_to_stack.append(hist[1])
    stacked_bins = torch.stack(bins_to_stack)
    min_edge = stacked_bins.min()
    max_edge = stacked_bins.max()
    # 2048 is the number of bins and 2049 is the number of edges
    merged_bins = torch.linspace(min_edge.item(), max_edge.item(), 2049)
    merged_counts = None

    for hist in module.hist:
        if merged_counts is None:
            merged_counts = _interpolate_hist(hist[0], hist[1], merged_bins)
        else:
            merged_counts += _interpolate_hist(hist[0], hist[1], merged_bins)

    module.hist = (merged_counts, merged_bins)


def _interpolate_hist(counts, bins, new_bins):
    """
    Helper function for interpolating histograms to new bins
    """
    cumulative_hist = torch.cumsum(counts, dim=0).to(device=bins.device)
    cumulative_hist = torch.cat((torch.tensor([0]), cumulative_hist))
    cumulative_interp_hist = torch.from_numpy(np.interp(new_bins.numpy(), bins.numpy(),
                                                        cumulative_hist.numpy()))
    interp_counts = torch.diff(cumulative_interp_hist, prepend=torch.tensor([0]))

    return interp_counts


# pylint: disable=unused-argument
def _hist_hook(module, inp, output):
    """
    Hook to collect histogram of activations
    """
    if not hasattr(module, 'hist'):
        module.hist = []
    # dynamic histogram collection
    hist = histogram(output.clone().detach().flatten(), bins=2048)
    module.hist.append(hist)


def register_hist_hooks(module):
    """
    Register hooks for histogram collection
    """
    module.handle = module.register_forward_hook(_hist_hook, always_call=True)


def release_hist_hooks(module):
    """
    Release hooks after histogram collection
    """
    module.handle.remove()


def _remove_outliers(module, outlier_removal_z_score=8.0):
    """
    Remove outliers from histogram
    """
    # Get mean and std of histogram
    hist_count = module.hist[0]
    hist_bins = module.hist[1]
    hist_bins_middle = []
    for i in range(len(hist_bins) - 1):
        hist_bins_middle.append((hist_bins[i] + hist_bins[i+1])/2)
    hist_bins_middle = torch.tensor(hist_bins_middle)
    mean = torch.sum(hist_count[1:] * hist_bins_middle) / torch.sum(hist_count[1:])
    std = torch.sqrt(torch.sum(hist_count[1:] * (hist_bins_middle - mean)**2)
                     / torch.sum(hist_count[1:]))

    # When activations are very small, std ends up being 0 due to rounding.
    # In this case, we set std to a very small value to prevent zero element histogram.
    if std == 0:
        std = 1e-9
    # Calculate bounds according to z-score
    upper_bound = mean + outlier_removal_z_score * std
    lower_bound = mean - outlier_removal_z_score * std
    hist_bins_middle = torch.cat((torch.tensor([0]), hist_bins_middle))
    # Remove outliers according to bounds
    hist_count[hist_bins_middle > upper_bound] = 0
    hist_count[hist_bins_middle < lower_bound] = 0
    non_zero_bins = hist_count != 0
    hist_count = hist_count[non_zero_bins]
    hist_bins = hist_bins[non_zero_bins]
    module.hist = (hist_count, hist_bins)


def init_threshold_module(module, outlier_removal_z_score):
    """
    Initialize activation threshold
    """
    _merge_hist(module)
    _remove_outliers(module, outlier_removal_z_score)
    module.activation_threshold = nn.Parameter(module.hist[1].abs().max().log2().ceil().exp2(),
                                               requires_grad=False)


def calc_threshold(module, iterations=5, bits=8):
    """
    Iteratively calculate threshold for activation quantization
    """
    e_min = torch.inf
    t_nc = module.activation_threshold
    t = None

    for i in range(iterations):
        t_i = t_nc / (2**i)
        e_i = calc_q_error(module, t_i, bits)
        if e_i < e_min:
            e_min = e_i
            t = t_i

    module.activation_threshold = nn.Parameter(torch.log2(t), requires_grad=False)


def calc_threshold_weight(weight, iterations=5, bits=8):
    """
    Iteratively calculate threshold for weight quantization
    """
    e_min = torch.inf
    outlier_removal_z_score = 3
    mean = weight.mean()
    std = weight.std()
    clamp_min = mean - outlier_removal_z_score * std
    clamp_max = mean + outlier_removal_z_score * std
    weight = weight.clamp(min=clamp_min, max=clamp_max)
    t_nc = weight.abs().max().log2().ceil().exp2()
    t = None
    qfunction = qSigned()
    best_i = 0
    for i in range(iterations):
        t_i = t_nc / (2**i)
        e_i = calc_q_error_weight(weight, t_i, bits, qfunction)
        if e_i < e_min:
            e_min = e_i
            best_i = i
            t = t_i
    print("best_i:", best_i)
    return t.log2().ceil()
    


class QuantizationAwareModule(nn.Module):
    """
    Common code for Quantization-Aware Training
    """
    def __init__(
            self,
            pooling=None,
            activation=None,
            wide=False,
            weight_bits=None,
            bias_bits=None,
            quantize_activation=False,
            pool=None,
            op=None,
            bn=None,
            shift_quantile=1.0,
            clamp_activation=False,
    ):
        super().__init__()

        assert weight_bits in [None, 1, 2, 4, 8], f'Weight bits cannot be {weight_bits}'
        assert bias_bits in [None, 1, 2, 4, 8], f'Bias bits cannot be {bias_bits}'

        self.quantize = None
        self.clamp = None
        self.quantize_bias = None
        self.clamp_bias = None
        self.calc_out_shift = None
        self.scale = None
        self.calc_weight_scale = None
        self.calc_out_scale = None
        self.quantize_weight = None
        self.clamp_weight = None
        self.quantize_pool = None
        self.clamp_pool = None

        self.activate = get_activation(activation)
        self.wide = wide

        self.pool = pool
        self.op = op
        if op is not None and not hasattr(self, '_conv_forward'):
            self._conv_forward = op._conv_forward  # pylint: disable=protected-access
        self.bn = bn
        self.pooling = pooling

        self.output_shift = nn.Parameter(torch.tensor([0.]), requires_grad=False)
   
        if dev.per_channel:
            out_channels = self.op.weight.shape[0]
            print("Out Channels", out_channels)
            self.alpha = nn.Parameter(torch.tensor(out_channels))
        else:
            self.alpha = nn.Parameter(torch.tensor(1.))
        # Activation threshold determined during QAT, used in quantization
        # It determines the range of quantization
        self.activation_threshold = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.final_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.init_module(weight_bits, bias_bits, quantize_activation,
                         clamp_activation, shift_quantile)

    def init_module(
            self,
            weight_bits,
            bias_bits,
            quantize_activation,
            clamp_activation,
            shift_quantile,
            export=False,
    ):
        """Initialize model parameters"""
        print("FP bias for testing")
        if weight_bits is None and bias_bits is None and not quantize_activation:
            if not export:
                self.weight_bits = nn.Parameter(torch.tensor([0]), requires_grad=False)
                self.bias_bits = nn.Parameter(torch.tensor([0]), requires_grad=False)
                self.quantize_activation = nn.Parameter(torch.tensor([False]), requires_grad=False)
                self.clamp_activation = nn.Parameter(torch.tensor([clamp_activation]),
                                                     requires_grad=False)
                self.adjust_output_shift = nn.Parameter(torch.tensor([False]), requires_grad=False)
        elif weight_bits in [1, 2, 4, 8] and bias_bits in [1, 2, 4, 8] and quantize_activation:
            self.weight_bits = nn.Parameter(torch.tensor([weight_bits]), requires_grad=False)
            if not export:
                self.bias_bits = nn.Parameter(torch.tensor([bias_bits]), requires_grad=False)
                self.quantize_activation = nn.Parameter(torch.tensor([True]), requires_grad=False)
                self.clamp_activation = nn.Parameter(torch.tensor([True]), requires_grad=False)
                self.adjust_output_shift = nn.Parameter(torch.tensor([not dev.simulate]),
                                                        requires_grad=False)
        else:
            assert False, f'Undefined mode with weight_bits: {weight_bits}, ' \
                          f'bias_bits: {bias_bits}, ' \
                          f'quantize_activation: {quantize_activation}'

        if not export:
            self.shift_quantile = nn.Parameter(torch.tensor([shift_quantile]), requires_grad=False)
            self.set_functions()

    def set_functions(self):
        """Set functions to be used wrt the model parameters"""
        if self.adjust_output_shift.detach():
            #print("Dev per channel", dev.per_channel)
            self.calc_out_shift = OutputShiftLimit(self.shift_quantile.detach().item(), dev.per_channel)
            self.calc_weight_scale = WeightScale()
        else:
            self.calc_out_shift = OutputShiftPassthrough()
            self.calc_weight_scale = One()
        
        self.scale = Scaler()
        self.calc_out_scale = OutputScale()

        self.quantize_weight, self.quantize_bias, self.clamp_weight, self.clamp_bias = \
            quantize_clamp_parameters(self.weight_bits.detach().item(),
                                      self.bias_bits.detach().item())
        self.quantize, self.clamp = \
            quantize_clamp(self.wide, bool(self.quantize_activation.detach().item()),
                           bool(self.clamp_activation.detach().item()),
                           int(self.weight_bits.detach().item()))
        self.quantize_pool, self.clamp_pool = \
            quantize_clamp_pool(self.pooling, bool(self.quantize_activation.detach().item()),
                                bool(self.clamp_activation.detach().item()))


        if dev.lsq_weight_scale and self.adjust_output_shift.detach() and self.op is not None:
            #self.alpha.data.copy_(2 * self.op.weight.abs().mean() / math.sqrt(2**self.weight_bits.detach().item() - 1))
            #params_r = torch.flatten(self.op.weight.detach())
            #self.alpha.data.copy_(torch.exp2(self.calc_out_shift(params_r, self.output_shift.detach())))
            #print("Alpha was", self.alpha)
            self.alpha.data.copy_(calc_threshold_weight(self.op.weight, iterations=5, bits=self.weight_bits.detach().item()))
            #self.alpha.data.copy_(self.calc_out_shift(params_r, self.output_shift.detach()))
            #print("Alpha now", self.alpha)
            #print("op weight numel", self.op.weight.numel())
            #self.g = 1.0 / math.sqrt(self.op.weight.numel() * (2**self.weight_bits.detach().item() - 1))
    
        self.FakeQuantizeAct = FakeQuantizeAct(self.activation_threshold.detach().item(),
                                               self.clamp)
        
        #FakeQuantizeActv2(self.activation_threshold.detach().item())
                            #
        self.FakeQuantizePool = FakeQuantizePool(self.clamp_pool)
        
        #FakeQuantizePoolv2()

        #self.FakeQuantizeWeight = FakeQuantizeWeightv2(self.weight_bits.detach().item(), dev.per_channel)#FakeQuantizeWeight(self.clamp_weight, self.weight_bits.detach().item(), dev.per_channel)
        self.FakeQuantizeWeight = qSigned()
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        if self.pool is not None:
            if not dev.fakeactquant:
            #if True:
                x = self.clamp_pool(self.quantize_pool(self.pool(x)))
            else:
                pool_scale = find_pool_scale(x)

                x = self.pool(x)
                x = self.FakeQuantizePool(x, pool_scale)
                #x = FakeGradModifierPool.apply(x, pool_scale, self.FakeQuantizePool)


        if self.op is not None:
            if self.op.bias is not None and not dev.fakeactquant:
                bias_r = torch.flatten(self.op.bias.detach())
                weight_r = torch.flatten(self.op.weight.detach())
                params_r = torch.cat((weight_r, bias_r))
            else:
                if not dev.per_channel:
                    params_r = torch.flatten(self.op.weight.detach())
                else:
                    params_r = self.op.weight.detach()
            if dev.lsq_weight_scale and self.adjust_output_shift.detach():
                pass
                #alpha = grad_scale(self.alpha, self.g)
                #Prevent alpha from going to 0
                #alpha = alpha.clamp(min=1e-9).log2()
                #out_shift = PoTAlpha.apply(alpha)
                #print("alpha grad", alpha.grad)
                #print("self.alpha.grad", self.alpha.grad)
                #print("Out Shift grad", out_shift.grad)
                #print("Out Shift", out_shift)
                #print("Alpha", alpha)
                #print("self.alpha", self.alpha)
                #print("Alpha", alpha)
            else:
                out_shift = self.calc_out_shift(params_r, self.output_shift.detach())
                weight_scale = self.calc_weight_scale(out_shift)

            
            if not dev.fakeactquant and not dev.lsq_weight_scale:
            #if True:
                x = self._conv_forward(  # pylint: disable=protected-access
                    x,
                    self.clamp_weight(self.quantize_weight(self.op.weight.mul(weight_scale))),
                    None if self.op.bias is None
                    else self.clamp_bias(self.quantize_bias(self.op.bias.mul(weight_scale))),
                    #self.op.bias,
                )
            elif not dev.fakeactquant and dev.lsq_weight_scale:
                x = self._conv_forward(  # pylint: disable=protected-access
                    x,
                    self.FakeQuantizeWeight.apply(self.op.weight, self.alpha, self.weight_bits.detach().item(), not dev.fakeactquant),
                    None if self.op.bias is None
                    else self.FakeQuantizeWeight.apply(self.op.bias, self.alpha.detach().clone(), self.bias_bits.detach().item(), not dev.fakeactquant),
                    #self.op.bias,
                )
            else:
                x = self._conv_forward(  # pylint: disable=protected-access
                    x,
                    self.FakeQuantizeWeight.apply(self.op.weight, self.alpha, self.weight_bits.detach().item()),
                    #self.FakeQuantizeWeight(self.op.weight, out_shift),
                    #FakeGradModifierWeight.apply(self.op.weight, out_shift, self.FakeQuantizeWeight),
                    self.op.bias,
                )

            if self.bn is not None:
                x = self.bn(x)
            # Quantized checkpoint will have subtracted threshold from output shift
            # Therefore, it shouldn't be done again in simulate mode
            if not dev.simulate and not dev.fakeactquant and not dev.lsq_weight_scale:
                #print("Out Shift", out_shift)
                #print("Activation Threshold", self.activation_threshold)
                out_shift = (out_shift - self.activation_threshold).clamp(min=-15., max=15.)

                out_scale = self.calc_out_scale(out_shift)
            elif not dev.simulate and not dev.fakeactquant and dev.lsq_weight_scale:
                # TODO: Check if this is correct
                #print("alpha", self.alpha.detach().clone().ceil())
                #print("activation threshold", self.activation_threshold)
                #print("activation threshold", self.activation_threshold)
                #print("self.alpha", self.alpha.detach().clone().ceil())
                #out_shift = (self.alpha.detach().clone().ceil() - self.activation_threshold).clamp(min=-15., max=15.)
                out_shift = (self.alpha.detach().clone().ceil()- self.activation_threshold).clamp(min=-15., max=15.)
                #print("Out Shift", out_shift)
                out_scale = self.calc_out_scale(out_shift)

            if not self.wide and not dev.fakeactquant:
                # The device does not apply output shift in wide mode
                x = self.scale(x, out_scale)
                #pass
                

            if not dev.fakeactquant:
            #if True:
                x = self.clamp(self.quantize(self.activate(x)))
                # This is the final scale for the output, in the device it will be realized in SW
                x = x.mul(2.**(self.final_scale))
            else:
                x = self.activate(x)
                x = self.FakeQuantizeAct(x)
                #x = FakeGradModifierAct.apply(x, self.FakeQuantizeAct)
        return x


class Conv2d(QuantizationAwareModule):
    """
    2D pooling ('Avg', 'Max' or None) optionally followed by
    2D convolution/transposed 2D convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(  # pylint: disable=too-many-arguments
            self,
            in_channels,
            out_channels,
            kernel_size,
            op='Conv2d',
            pooling=None,
            pool_size=2,
            pool_stride=2,
            pool_dilation=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            activation=None,
            wide=False,
            batchnorm=None,
            weight_bits=None,
            bias_bits=None,
            quantize_activation=False,
            groups=1,
            eps=1e-05,
            momentum=0.05,
    ):
        assert not wide or activation is None
        #print(f"stride: {stride}")
        if pooling is not None:
            if pool_stride is None:
                pool_stride = pool_size

            if isinstance(pool_size, int):
                assert dev.device != 84 or pool_size & 1 == 0
                assert pool_size <= 16 \
                    and (dev.device != 84 or pool_size <= 4 or pooling == 'Max')
            elif isinstance(pool_size, tuple):
                assert len(pool_size) == 2
                assert dev.device != 84 or pool_size[0] & 1 == 0
                assert pool_size[0] <= 16 \
                    and (dev.device != 84 or pool_size[0] <= 4 or pooling == 'Max')
                assert dev.device != 84 or pool_size[1] & 1 == 0
                assert pool_size[1] <= 16 \
                    and (dev.device != 84 or pool_size[1] <= 4 or pooling == 'Max')
            else:
                raise ValueError('pool_size must be int or tuple')

            if isinstance(pool_stride, int):
                assert pool_stride > 0
                assert pool_stride <= 16 \
                    and (dev.device != 84 or pool_stride <= 4 or pooling == 'Max')
            elif isinstance(pool_stride, tuple):
                assert len(pool_stride) == 2
                assert dev.device != 84 or pool_stride[0] == pool_stride[1]
                assert 0 < pool_stride[0] <= 16 \
                    and (dev.device != 84 or pool_stride[0] <= 4 or pooling == 'Max')
                assert 0 < pool_stride[1] <= 16 \
                    and (dev.device != 84 or pool_stride[1] <= 4 or pooling == 'Max')
                assert pool_stride[0] == pool_stride[1]
            else:
                raise ValueError('pool_stride must be int or tuple')

            if isinstance(pool_dilation, int):
                assert pool_dilation > 0
                assert pool_dilation <= 1 \
                    or dev.device == 87 and pool_dilation <= 16 and pooling == 'Max'
            elif isinstance(pool_dilation, tuple):
                assert len(pool_dilation) == 2
                assert pool_dilation[0] > 0
                assert pool_dilation[0] <= 1 \
                    or dev.device == 87 and pool_dilation[0] <= 16 and pooling == 'Max'
                assert pool_dilation[1] > 0
                assert pool_dilation[1] <= 1 \
                    or dev.device == 87 and pool_dilation[1] <= 16 and pooling == 'Max'
            else:
                raise ValueError('pool_dilation must be int or tuple')

            if op == 'ConvTranspose2d':
                assert stride == 2
            else:
                assert stride == 1
        else:
            if op == 'ConvTranspose2d':
                assert stride == 2
            else:
                assert 0 < stride <= 3

        #assert 0 <= padding <= 2
        if padding < 0 or padding > 2:
            print("Warning: Allowing padding < 0 or padding > 2 for ResNet tests")
        #assert dilation == 1
        if dilation != 1:
            print("Warning: Allowing dilation != 1 for ResNet tests")

        if pooling == 'Max':
            print("Warning: Allowing padding =! 0 for pooling")
            pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride,
                                dilation=pool_dilation, padding=padding)
        elif pooling == 'Avg':
            pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride, padding=0)
        else:
            pool = None

        if batchnorm == 'Affine':
            bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=True)
            #assert bias, '`bias` must be set (enable --use-bias for models where bias is optional)'
            if not bias:
                print("Warning: Allowing no bias batchnorm for testing")
        elif batchnorm == 'NoAffine':
            bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=False)
            assert bias, '`bias` must be set (enable --use-bias for models where bias is optional)'
        else:
            bn = None

        if kernel_size is not None:
            if isinstance(kernel_size, tuple):
                assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
                kernel_size = kernel_size[0]

            #assert kernel_size == 3 or dev.device != 84 and kernel_size == 1
            if kernel_size != 3 and (dev.device == 84 or kernel_size != 1):
                print("Warning: Allowing kernel size != 3 or 1 for ResNet tests")

            #assert groups == 1 or dev.device == 87, 'Set device to MAX78002 for depthwise support'
            if groups != 1 and dev.device != 87:
                print("Warning: Allowing groups != 1 for ResNet tests")

            if op == 'Conv2d':
                opn = nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias, groups=groups)
            elif op == 'ConvTranspose2d':
                assert dev.device != 84
                opn = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         output_padding=1, padding=padding,
                                         dilation=dilation, bias=bias, groups=groups)
            else:
                raise ValueError('Unsupported operation')
        else:
            opn = None

        super().__init__(
            pooling,
            activation,
            wide,
            weight_bits,
            bias_bits,
            quantize_activation,
            pool,
            opn,
            bn,
        )


class FusedMaxPoolConv2d(Conv2d):
    """
    Fused 2D Max Pool, 2D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConv2dBN(FusedMaxPoolConv2d):
    """
    Fused 2D Max Pool, 2D Convolution, BatchNorm and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConv2dReLU(FusedMaxPoolConv2d):
    """
    Fused 2D Max Pool, 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConv2dBNReLU(FusedMaxPoolConv2dReLU):
    """
    Fused 2D Max Pool, 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConv2dAbs(FusedMaxPoolConv2d):
    """
    Fused 2D Max Pool, 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedMaxPoolConv2dBNAbs(FusedMaxPoolConv2dAbs):
    """
    Fused 2D Max Pool, 2D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class MaxPool2d(FusedMaxPoolConv2d):
    """
    2D Max Pool
    """
    def __init__(self, kernel_size, stride=None, dilation=1, **kwargs):
        super().__init__(0, 0, None, pool_size=kernel_size, pool_stride=stride,
                         pool_dilation=dilation, activation=None, **kwargs)


class FusedAvgPoolConv2d(Conv2d):
    """
    Fused 2D Avg Pool, 2D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Avg', **kwargs)


class FusedAvgPoolConv2dReLU(FusedAvgPoolConv2d):
    """
    Fused 2D Avg Pool, 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedAvgPoolConv2dBNReLU(FusedAvgPoolConv2dReLU):
    """
    Fused 2D Avg Pool, 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedAvgPoolConv2dAbs(FusedAvgPoolConv2d):
    """
    Fused 2D Avg Pool, 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedAvgPoolConv2dBNAbs(FusedAvgPoolConv2dAbs):
    """
    Fused 2D Avg Pool, 2D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class AvgPool2d(FusedAvgPoolConv2d):
    """
    2D Avg Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__(0, 0, None, pool_size=kernel_size, pool_stride=stride,
                         activation=None, **kwargs)


class FusedConv2dReLU(Conv2d):
    """
    Fused 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedConv2dReLU6(Conv2d):
    """
    Fused 2D Convolution and ReLU6
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU6', **kwargs)


class FusedConv2dBN(Conv2d):
    """
    Fused 2D Convolution and BatchNorm
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedConv2dBNReLU(FusedConv2dReLU):
    """
    Fused 2D Convolution and BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedConv2dBNReLU6(FusedConv2dReLU6):
    """
    Fused 2D Convolution and BatchNorm and ReLU6
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedConv2dAbs(Conv2d):
    """
    Fused 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class DepthwiseConv2d(Conv2d):
    """
    AI8X - Fused 2D Depthwise Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedDepthwiseConv2dReLU(FusedConv2dReLU):
    """
    AI8X - Fused 2D Depthwise Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedDepthwiseConv2dBNReLU(FusedConv2dBNReLU):
    """
    AI8X - Fused 2D Convolution and BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedDepthwiseConv2dBNReLU6(FusedConv2dBNReLU6):
    """
    AI8X - Fused 2D Convolution and BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedAvgPoolDepthwiseConv2d(FusedAvgPoolConv2d):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and no activation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedAvgPoolDepthwiseConv2dReLU(FusedAvgPoolConv2dReLU):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedAvgPoolDepthwiseConv2dBNReLU(FusedAvgPoolConv2dBNReLU):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedMaxPoolDepthwiseConv2d(FusedMaxPoolConv2d):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and no activation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedMaxPoolDepthwiseConv2dReLU(FusedMaxPoolConv2dReLU):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedMaxPoolDepthwiseConv2dBNReLU(FusedMaxPoolConv2dBNReLU):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class ConvTranspose2d(Conv2d):
    """
    2D pooling ('Avg', 'Max' or None) optionally followed by
    transposed 2D convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, op='ConvTranspose2d', **kwargs)

    def _conv_forward(self, x, weight, bias):  # pylint: disable=method-hidden
        output_padding = self.op._output_padding(  # pylint: disable=protected-access
            x, None, self.op.stride, self.op.padding,  # type: ignore[arg-type]
            self.op.kernel_size, 2, self.op.dilation)  # type: ignore[arg-type]

        return nn.functional.conv_transpose2d(  # pylint: disable=method-hidden
            x, weight, bias, self.op.stride, self.op.padding,
            output_padding, self.op.groups, self.op.dilation)


class FusedConvTranspose2dReLU(ConvTranspose2d):
    """
    Fused Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedConvTranspose2dAbs(ConvTranspose2d):
    """
    Fused Transposed 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedConvTranspose2dBNReLU(FusedConvTranspose2dReLU):
    """
    Fused Transposed 2D Convolution and BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConvTranspose2d(ConvTranspose2d):
    """
    Fused 2D Max Pool, Transposed 2D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConvTranspose2dReLU(FusedMaxPoolConvTranspose2d):
    """
    Fused 2D Max Pool, Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConvTranspose2dBNReLU(FusedMaxPoolConvTranspose2dReLU):
    """
    Fused 2d Max Pool, Transposed 2D Convolution and BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConvTranspose2dAbs(FusedMaxPoolConvTranspose2d):
    """
    Fused 2D Max Pool, Transposed 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedAvgPoolConvTranspose2d(ConvTranspose2d):
    """
    Fused 2D Avg Pool, Transposed 2D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Avg', **kwargs)


class FusedAvgPoolConvTranspose2dReLU(FusedAvgPoolConvTranspose2d):
    """
    Fused 2D Avg Pool, Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedAvgPoolConvTranspose2dBNReLU(FusedAvgPoolConvTranspose2dReLU):
    """
    Fused 2d Avg Pool, Transposed 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedAvgPoolConvTranspose2dAbs(FusedAvgPoolConvTranspose2d):
    """
    Fused 2D Avg Pool, Transposed 2D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class DepthwiseConvTranspose2d(ConvTranspose2d):
    """
    AI8X - Depthwise Transposed 2D Convolution
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedDepthwiseConvTranspose2dReLU(FusedConvTranspose2dReLU):
    """
    AI8X - Fused Depthwise Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedDepthwiseConvTranspose2dBNReLU(FusedConvTranspose2dBNReLU):
    """
    AI8X - Fused Depthwise Transposed 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedAvgPoolDepthwiseConvTranspose2d(FusedAvgPoolConvTranspose2d):
    """
    AI8X - Fused 2D Avg Pool, Depthwise Transposed 2D Convolution and no activation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedAvgPoolDepthwiseConvTranspose2dReLU(FusedAvgPoolConvTranspose2dReLU):
    """
    AI8X - Fused 2D Avg Pool, Depthwise Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedAvgPoolDepthwiseConvTranspose2dBNReLU(FusedAvgPoolConvTranspose2dBNReLU):
    """
    AI8X - Fused 2D Avg Pool, Depthwise Transposed 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedMaxPoolDepthwiseConvTranspose2d(FusedMaxPoolConvTranspose2d):
    """
    AI8X - Fused 2D Max Pool, Depthwise Transposed 2D Convolution and no activation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedMaxPoolDepthwiseConvTranspose2dReLU(FusedMaxPoolConvTranspose2dReLU):
    """
    AI8X - Fused 2D Max Pool, Depthwise Transposed 2D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedMaxPoolDepthwiseConvTranspose2dBNReLU(FusedMaxPoolConvTranspose2dBNReLU):
    """
    AI8X - Fused 2D Max Pool, Depthwise Transposed 2D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, groups=args[0], **kwargs)


class FusedSoftwareLinearReLU(nn.Module):
    """
    Fused Linear and ReLU using Software
    """
    def __init__(self, in_features, out_features, bias=None, relu=True):
        super().__init__()

        if dev.device != 84:
            print('WARNING: SoftwareLinear should be used on AI84 only')

        self.op = nn.Linear(in_features, out_features, bias is True)  # False or None -> False

        if dev.simulate:
            self.quantize = Quantize(num_bits=dev.DATA_BITS)
            bits = dev.FC_ACTIVATION_BITS
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize = Empty()
            self.clamp = Clamp(min_val=-1., max_val=127./128.)  # Do not combine with ReLU

        if relu:
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = Empty()

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.op(x)
        x = self.clamp(self.quantize(self.activate(x)))
        return x


class SoftwareLinear(FusedSoftwareLinearReLU):
    """
    Linear using Software
    """
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, relu=False, **kwargs)


class Linear(QuantizationAwareModule):
    """
    Fused Linear and activation ('ReLU', 'Abs', None)
    """
    def __init__(
            self,
            in_features,
            out_features,
            pooling=None,
            bias=None,
            activation=None,
            wide=False,
            batchnorm=None,  # pylint: disable=unused-argument
            weight_bits=None,
            bias_bits=None,
            quantize_activation=False,
    ):
        assert not wide or activation is None

        assert dev.device != 84
        #assert in_features <= 1024
        if in_features > 1024:
            print(f'WARNING: in_features {in_features} > 1024')
        assert out_features <= 1024
        assert pooling is None
        assert batchnorm is None

        super().__init__(
            pooling,
            activation,
            wide,
            weight_bits,
            bias_bits,
            quantize_activation,
            None,
            nn.Linear(in_features, out_features, bias is True),
            None,
        )

        # Define dummy arguments to make Linear and Conv1d/Conv2d compatible.
        self.op.stride = None
        self.op.padding = None
        self.op.dilation = None
        self.op.groups = None

    def _conv_forward(self, x, weight, bias):  # pylint: disable=method-hidden
        return nn.functional.linear(x, weight, bias)


class FusedLinearReLU(Linear):
    """
    Fused Linear and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedLinearAbs(Linear):
    """
    Fused Linear and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class Conv1d(QuantizationAwareModule):
    """
    Fused 1D Pool ('Avg', 'Max' or None) followed by
    1D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(  # pylint: disable=too-many-arguments
            self,
            in_channels,
            out_channels,
            kernel_size,
            pooling=None,
            pool_size=2,
            pool_stride=2,
            pool_dilation=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            activation=None,
            wide=False,
            batchnorm=None,
            weight_bits=None,
            bias_bits=None,
            quantize_activation=False,
            groups=1,
            eps=1e-05,
            momentum=0.05,
    ):
        assert not wide or activation is None

        if pooling is not None:
            if pool_stride is None:
                pool_stride = pool_size

            assert dev.device != 84 or pool_size & 1 == 0
            assert pool_size <= 16 \
                and (dev.device != 84 or pool_size <= 4 or pooling == 'Max')

            assert 0 < pool_stride <= 16 \
                and (dev.device != 84 or pool_stride <= 4 or pooling == 'Max')

            assert pool_dilation > 0
            assert pool_dilation <= 1 \
                or dev.device == 87 and pool_dilation <= 16 and pooling == 'Max'

            assert stride == 1
        else:
            assert dev.device != 84 or stride == 3
            assert dev.device == 84 or stride == 1

        if pooling == 'Max':
            pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride,
                                dilation=pool_dilation, padding=0)
        elif pooling == 'Avg':
            pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride, padding=0)
        else:
            pool = None

        if batchnorm == 'Affine':
            bn = nn.BatchNorm1d(out_channels, eps=eps, momentum=momentum, affine=True)
            assert bias, '`bias` must be set (enable --use-bias for models where bias is optional)'
        elif batchnorm == 'NoAffine':
            bn = nn.BatchNorm1d(out_channels, eps=eps, momentum=momentum, affine=False)
            assert bias, '`bias` must be set (enable --use-bias for models where bias is optional)'
        else:
            bn = None

        if kernel_size is not None:
            assert dev.device != 84 or padding in [0, 3, 6]
            assert dev.device == 84 or padding in [0, 1, 2]
            assert dev.device != 84 or kernel_size == 9
            assert dev.device == 84 or kernel_size in [1, 2, 3, 4, 5, 6, 7, 8, 9]

            assert (kernel_size - 1) * dilation < 9 or padding == 0 and kernel_size <= 3

            assert groups == 1 or dev.device == 87, 'Set device to MAX78002 for depthwise support'

            assert padding == 0 or in_channels <= 64 or dev.device != 87, \
                'This device requires pad==0 when using more than 64 input channels in Conv1d'

            opn = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=bias, groups=groups)
        else:
            opn = None

        super().__init__(
            pooling,
            activation,
            wide,
            weight_bits,
            bias_bits,
            quantize_activation,
            pool,
            opn,
            bn,
        )


class FusedMaxPoolConv1d(Conv1d):
    """
    Fused 1D Max Pool, 1D Convolution and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Max', **kwargs)


class FusedMaxPoolConv1dBN(FusedMaxPoolConv1d):
    """
    Fused 1D Max Pool, 1D Convolution, BatchNorm and Activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConv1dReLU(FusedMaxPoolConv1d):
    """
    Fused 1D Max Pool, 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedMaxPoolConv1dBNReLU(FusedMaxPoolConv1dReLU):
    """
    Fused 1D Max Pool, 1D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedMaxPoolConv1dAbs(FusedMaxPoolConv1d):
    """
    Fused 1D Max Pool, 1D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedMaxPoolConv1dBNAbs(FusedMaxPoolConv1d):
    """
    Fused 1D Max Pool, 1D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class MaxPool1d(FusedMaxPoolConv1d):
    """
    1D Max Pool
    """
    def __init__(self, kernel_size, stride=None, dilation=1, **kwargs):
        super().__init__(0, 0, None, pool_size=kernel_size, pool_stride=stride,
                         pool_dilation=dilation, activation=None, **kwargs)


class FusedAvgPoolConv1d(Conv1d):
    """
    Fused 1D Avg Pool, 1D Convolution and activation ('ReLU', 'Abs', None)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pooling='Avg', **kwargs)


class FusedAvgPoolConv1dReLU(FusedAvgPoolConv1d):
    """
    Fused 1D Avg Pool, 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedAvgPoolConv1dBNReLU(FusedAvgPoolConv1dReLU):
    """
    Fused 1D Avg Pool, 1D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedAvgPoolConv1dAbs(FusedAvgPoolConv1d):
    """
    Fused 1D Avg Pool, 1D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedAvgPoolConv1dBNAbs(FusedAvgPoolConv1d):
    """
    Fused 1D Avg Pool, 1D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class AvgPool1d(FusedAvgPoolConv1d):
    """
    1D Avg Pool
    """
    def __init__(self, kernel_size, stride=None, **kwargs):
        super().__init__(0, 0, None, pool_size=kernel_size, pool_stride=stride,
                         activation=None, **kwargs)


class FusedConv1dReLU(Conv1d):
    """
    Fused 1D Convolution and ReLU
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='ReLU', **kwargs)


class FusedConv1dBNReLU(FusedConv1dReLU):
    """
    Fused 1D Convolution, BatchNorm and ReLU
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class FusedConv1dAbs(Conv1d):
    """
    Fused 1D Convolution and Abs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation='Abs', **kwargs)


class FusedConv1dBNAbs(FusedConv1dAbs):
    """
    Fused 1D Convolution, BatchNorm and Abs
    """
    def __init__(self, *args, **kwargs):
        if 'batchnorm' not in kwargs:
            kwargs['batchnorm'] = 'Affine'
        super().__init__(*args, **kwargs)


class Eltwise(nn.Module):
    """
    Base Class for Elementwise Operation
    """
    def __init__(self, f, clamp_activation=False):
        super().__init__()
        self.f = f
        self.activation_threshold = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.set_clamp(clamp_activation)

    def set_clamp(self, clamp_activation):
        """
        Set Clamping Function
        """
        if dev.simulate or dev.fakeactquant:
            bits = dev.ACTIVATION_BITS
            #self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
            self.FakeQuantizeAct =  FakeQuantizeActv2(self.activation_threshold.detach().item())
            #FakeQuantizeAct(self.activation_threshold.detach().item(),
                                    #           self.clamp)

        else:
            if clamp_activation:
                self.clamp = Clamp(min_val=-1., max_val=127./128.)
            else:
                self.clamp = Empty()

    def forward(self, *x):
        """Forward prop"""
        y = x[0]
        for i in range(1, len(x)):
            y = self.f(y, x[i])
        if dev.fakeactquant:
            x = self.FakeQuantizeAct(y)
            #x = FakeGradModifierAct.apply(y, self.FakeQuantizeAct)
            #y = QuantizationFunction.apply(y, dev.ACTIVATION_BITS, self.activation_threshold)
            #y = self.clamp(y)
            #x = DequantizationFunction.apply(y, dev.ACTIVATION_BITS, self.activation_threshold)
        else:
            x = self.clamp(y)
        return x


class ActivationQHolder(Eltwise):
    """
    Activation Quantization Holder
    """
    @staticmethod
    def shift(x, activation_threshold):
        """
        Shift Activation by Activation Threshold
        """
        if dev.fakeactquant:
            return x
        #print("ActivationQHolder shift", x.min(), x.max(), activation_threshold)
        x /= (2**activation_threshold)
        #print("ActivationQHolder shift", x.min(), x.max())
        return x

    def __init__(self):
        super().__init__(self.shift)


class Add(Eltwise):
    """
    Elementwise Add Operation
    """
    def __init__(self):
        super().__init__(torch.add)


class Sub(Eltwise):
    """
    Elementwise Subtract Operation
    """

    @staticmethod
    def sub(a, b):
        """
        Subtract Tensors
        """
        return torch.add(a, torch.neg(b))

    def __init__(self):
        super().__init__(self.sub)


class BitwiseXor(Eltwise):
    """
    Elementwise Bitwise Xor Operation
    """

    @staticmethod
    def bitwise_xor(a, b):
        """
        Bitwise XOR of Tensors via int intermediate
        """
        # Convert input from float to byte
        a = a.add(.5).mul(256.).round().int()
        b = b.add(.5).mul(256.).round().int()
        # Bitwise XOR on integers, convert back to float
        return torch.bitwise_xor(a, b).div(256.).sub(.5)

    def __init__(self):
        super().__init__(self.bitwise_xor)


class BitwiseOr(Eltwise):
    """
    Elementwise Bitwise Or Operation
    """

    @staticmethod
    def bitwise_or(a, b):
        """
        Bitwise OR of Tensors via int intermediate
        """
        a = a.add(.5).mul(256.).round().int()
        b = b.add(.5).mul(256.).round().int()
        # Bitwise OR on integers, convert back to float
        return torch.bitwise_or(a, b).div(256.).sub(.5)

    def __init__(self):
        super().__init__(self.bitwise_or)


class Device:
    """
    Device base class
    """
    def __init__(self, device, simulate, round_avg, fakeactquant=False, per_channel=False, lsq_weight_scale=False):
        self.device = device
        self.simulate = simulate
        self.round_avg = round_avg
        self.fakeactquant = fakeactquant
        self.per_channel = per_channel
        self.lsq_weight_scale = lsq_weight_scale
        print(f'WARNING: lsq_weight_scale={lsq_weight_scale}')

    def __str__(self):
        return self.__class__.__name__


class DevAI84(Device):
    """
    Implementation limits for AI84
    """
    def __init__(self, simulate, round_avg, fakeactquant=False, per_channel=False, lsq_weight_scale=False):
        assert not round_avg

        super().__init__(84, simulate, round_avg, fakeactquant, per_channel, lsq_weight_scale)

        self.WEIGHT_BITS = 8
        self.DATA_BITS = 8
        self.ACTIVATION_BITS = 8
        self.FULL_ACC_BITS = 8
        self.FC_ACTIVATION_BITS = 16
        self.WIDE_LAYER_RESOLUTION_BITS = 8

        self.WEIGHT_INPUTS = 64
        self.WEIGHT_DEPTH = 128

        self.MAX_AVG_POOL = 4

    def __str__(self):
        return self.__class__.__name__


class DevAI85(Device):
    """
    Implementation limits for MAX78000
    """
    def __init__(self, simulate, round_avg, fakeactquant=False, per_channel=False, lsq_weight_scale=False):
        super().__init__(85, simulate, round_avg, fakeactquant, per_channel, lsq_weight_scale)

        self.WEIGHT_BITS = 8
        self.DATA_BITS = 8
        self.ACTIVATION_BITS = 8
        self.FULL_ACC_BITS = 30
        self.FC_ACTIVATION_BITS = 16
        self.WIDE_LAYER_RESOLUTION_BITS = 15

        self.WEIGHT_INPUTS = 256
        self.WEIGHT_DEPTH = 768

        self.MAX_AVG_POOL = 16

    def __str__(self):
        return self.__class__.__name__


class DevAI87(Device):
    """
    Implementation limits for MAX78002.
    """
    def __init__(self, simulate, round_avg, fakeactquant=False, per_channel=False, lsq_weight_scale=False):
        super().__init__(87, simulate, round_avg, fakeactquant, per_channel, lsq_weight_scale)

        self.WEIGHT_BITS = 8
        self.DATA_BITS = 8
        self.ACTIVATION_BITS = 8
        self.FULL_ACC_BITS = 30
        self.FC_ACTIVATION_BITS = 16
        self.WIDE_LAYER_RESOLUTION_BITS = 15

        self.WEIGHT_INPUTS = 256
        self.WEIGHT_DEPTH = 5120

        self.MAX_AVG_POOL = 16

    def __str__(self):
        return self.__class__.__name__


def set_device(
        device,
        simulate,
        round_avg,
        fakeactquant=False,
        per_channel=False,
        lsq_weight_scale=False,
        verbose=True,
):
    """
    Change implementation configuration to match the `device` input value and
    `simulate` bool. `round_avg` controls the average pooling rounding.
    """
    global dev  # pylint: disable=global-statement

    if verbose:
        print(f'Configuring device: {devices.partnum(device)}, simulate={simulate}, fakeactquant={fakeactquant}, per_channel={per_channel}, lsq_weight_scale={lsq_weight_scale}')

    if device == 84:
        dev = DevAI84(simulate, round_avg, fakeactquant, per_channel, lsq_weight_scale)
    elif device == 85:
        dev = DevAI85(simulate, round_avg, fakeactquant, per_channel, lsq_weight_scale)
    elif device == 87:
        dev = DevAI87(simulate, round_avg, fakeactquant, per_channel, lsq_weight_scale)
    else:
        raise ValueError(f'Unkown device {device}.')


class QuantizeONNX(nn.Module):
    """
    Post-activation integer quantization module
    Apply the custom autograd function
    """
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        factor = 2**(self.num_bits-1)
        return x.mul(factor).round().div(factor)


def initiate_qat(m, qat_policy, export=False):
    """
    Modify the (unwrapped) model `m` to start quantization aware training.
    """
    for name, module in m.named_modules():
        if isinstance(module, QuantizationAwareModule) and hasattr(module, 'weight_bits'):
            if 'shift_quantile' in qat_policy:
                print(f'Initiating {name} with {qat_policy["weight_bits"]} weight bits, '
                      f'{qat_policy["bias_bits"]} bias bits, shift_quantile={qat_policy["shift_quantile"]}')
                module.init_module(qat_policy['weight_bits'],
                                   qat_policy['bias_bits'],
                                   True, True, qat_policy['shift_quantile'], export)
            else:
                print(f'Initiating {name} with {qat_policy["weight_bits"]} weight bits, '
                      f'{qat_policy["bias_bits"]} bias bits, shift_quantile=1.0')
                module.init_module(qat_policy['weight_bits'],
                                   qat_policy['bias_bits'], True, True, 1.0, export)
            if 'overrides' in qat_policy:
                if name in qat_policy['overrides']:
                    if 'weight_bits' in qat_policy['overrides'][name]:
                        weight_field = qat_policy['overrides'][name]['weight_bits']
                    else:
                        weight_field = qat_policy['weight_bits']
                    if 'shift_quantile' in qat_policy['overrides'][name]:
                        print(f'Overriding {name} with {weight_field} weight bits, '
                            f'shift_quantile={qat_policy["overrides"][name]["shift_quantile"]}')
                        module.init_module(weight_field, weight_field, True,
                                           True, qat_policy['overrides'][name]['shift_quantile'],
                                           export)
                    elif 'shift_quantile' in qat_policy:
                        print(f'Overriding {name} with {weight_field} weight bits, '
                            f'shift_quantile={qat_policy["shift_quantile"]}')
                        module.init_module(weight_field, weight_field, True,
                                           True, qat_policy['shift_quantile'], export)
                    else:
                        print(f'Overriding {name} with {weight_field} weight bits, '
                            f'shift_quantile=1.0')
                        module.init_module(weight_field,
                                           weight_field, True, True, 1.0, export)
        elif isinstance(module, Eltwise):
            module.set_clamp(True)


def update_model(m):
    """
    Update model `m` with the current parameters.
    It is used to update model functions after loading a checkpoint file.
    """
    for _, module in m.named_modules():
        if isinstance(module, QuantizationAwareModule):
            module.set_functions()


def update_optimizer(m, optimizer):
    """
    Update optimizer after model 'm' had a batchnorm fusion.
    This is needed to update the optimizer state_dict to match the new model parameters.
    """
    old_state_dict = optimizer.state_dict()
    old_groups = optimizer.param_groups
    optimizer = type(optimizer)(m.parameters(), **optimizer.defaults)
    new_state_dict = optimizer.state_dict()
    groups = optimizer.param_groups

    for x, g in enumerate(groups):
        key_reduce = 0
        for p in g['params']:
            if (len(p.shape) == 1 and p.shape[0] == 1):
                continue
            nf_keys = []
            for key in old_state_dict['state'].keys():
                sub_keys = old_state_dict['state'][key].keys()
                if old_groups[x]['params'][int(key)].shape == p.shape:
                    for y, sub_key in enumerate(sub_keys):
                        if y == 0:
                            new_state_dict['state'][key-key_reduce] = \
                                {sub_key: old_state_dict['state'][key][sub_key]}
                        else:
                            new_state_dict['state'][key-key_reduce][sub_key] = \
                                old_state_dict['state'][key][sub_key]
                    old_state_dict['state'].pop(key)
                    break
                nf_keys.append(key)
                key_reduce += 1
            for key in nf_keys:
                old_state_dict['state'].pop(key)
        for key in old_state_dict['param_groups'][x].keys():
            if key != 'params':
                new_state_dict['param_groups'][x][key] = \
                    old_state_dict['param_groups'][x][key]

    optimizer.load_state_dict(new_state_dict)
    return optimizer


def fuse_bn_layers(m):
    """
    Fuse the bn layers before the quantization aware training starts.
    """
    for _, module in m.named_modules():
        if isinstance(module, QuantizationAwareModule) and module.bn is not None:
            w = module.op.weight.data
            b = module.op.bias.data
            device = w.device

            r_mean = module.bn.running_mean
            r_var = module.bn.running_var
            r_inv_std = torch.rsqrt(r_var + module.bn.eps)
            beta = module.bn.weight
            gamma = module.bn.bias

            if beta is None:
                beta = torch.ones(w.shape[0], device=device)
            if gamma is None:
                gamma = torch.zeros(w.shape[0], device=device)

            w_new = w * (beta * r_inv_std).reshape((w.shape[0],) + (1,) * (len(w.shape) - 1))
            b_new = (b - r_mean) * r_inv_std * beta + gamma

            module.op.weight.data = w_new
            module.op.bias.data = b_new
            module.bn = None


def apply_scales(model):
    """
    Readjust the scales and apply according to the model graph.
    """
    net_graph = symbolic_trace(model)
    adds = {}
    concats = {}
    prevs = {}
    op_names = ["torch.conv2d", "torch.conv1d", "torch.linear",
                "torch._C._nn.linear", "torch.conv_transpose2d"]
    nodes_to_search = []
    name_prev = None
    print("Model graph: ", net_graph.graph)
    # Model graph traversal for finding the adds, concats and previous layers
    for node in net_graph.graph.nodes:
        name = node.format_node()
        if ("torch.add" in name) or ("torch.cat" in name):
            print("Node: ", name)
            nodes_to_search.clear()
            if "target=view" in name:
                if len(node.all_input_nodes) > 0:
                    input_node = (node.all_input_nodes)[0]
                    nodes_to_search.append(input_node)
            else:
                nodes_to_search.extend(node.all_input_nodes)
            for node_prev in reversed(net_graph.graph.nodes):
                name_prev = node_prev.format_node()
                if any(op_name in name_prev for op_name in op_names):
                    if node_prev in nodes_to_search:
                        node_prev_name = next(reversed(node_prev.__dict__['meta']
                                                       ['nn_module_stack']))
                        if "torch.add" in name:
                            node_name = next(reversed(node.__dict__['meta']['nn_module_stack']))
                            adds[node_prev_name] = node_name
                        elif "torch.cat" in name:
                            concats[node_prev_name] = str(node)
                        nodes_to_search.pop(nodes_to_search.index(node_prev))
                else:
                    if node_prev in nodes_to_search:
                        nodes_to_search.pop(nodes_to_search.index(node_prev))
                        if "target=view" in name_prev:
                            if len(node_prev.all_input_nodes) > 0:
                                input_node = (node_prev.all_input_nodes)[0]
                                nodes_to_search.append(input_node)
                        else:
                            nodes_to_search.extend(node_prev.all_input_nodes)

        elif any(op_name in name for op_name in op_names):
            nodes_to_search.clear()
            if len(node.all_input_nodes) > 0:
                input_node = (node.all_input_nodes)[0]
                nodes_to_search.append(input_node)
            for node_prev in reversed(net_graph.graph.nodes):
                name_prev = node_prev.format_node()
                if any((op_name in name_prev or "operator.truediv" in name_prev) for op_name in op_names):
                    if node_prev in nodes_to_search:
                        node_prev_name = next(reversed(node_prev.__dict__['meta']
                                                       ['nn_module_stack']))
                        node_name = next(reversed(node.__dict__['meta']['nn_module_stack']))
                        if prevs.get(str(node_name)) is None:
                            node_prevs = []
                            node_prevs.append(str(node_prev_name))
                            prevs[str(node_name)] = node_prevs
                        else:
                            prevs[str(node_name)].append(str(node_prev_name))
                        nodes_to_search.pop(nodes_to_search.index(node_prev))

                else:
                    for name_node in nodes_to_search:
                        if node_prev == name_node:
                            nodes_to_search.extend(node_prev.all_input_nodes)
                            nodes_to_search.pop(nodes_to_search.index(name_node))

    # Override the thresholds of layers that are connected to adds
    for name, module in model.named_modules():
        if isinstance(module, QuantizationAwareModule):
            if name in adds:
                for name1, module1 in model.named_modules():
                    if isinstance(module1, Eltwise):
                        if adds[name] == name1:
                            module.activation_threshold = module1.activation_threshold
                            break

    # Find the maximum threshold from the layers that are concatenated together
    concat_thresholds = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizationAwareModule):
            if name in concats:
                if concat_thresholds.get(concats[name]) is None:
                    concat_thresholds[concats[name]] = module.activation_threshold
                elif module.activation_threshold > concat_thresholds[concats[name]]:
                    concat_thresholds[concats[name]] = module.activation_threshold

    # Apply the maximum threshold to the layers that are concatenated together
    for name, module in model.named_modules():
        if isinstance(module, QuantizationAwareModule):
            if name in concats:
                module.activation_threshold = nn.Parameter(concat_thresholds[concats[name]],
                                                           requires_grad=False)

    # Find weight sharing layers and apply the maximum threshold from the multiple passes
    shared_threshold = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizationAwareModule):
            if prevs.get(name) is not None:
                for prev in prevs[name]:
                    for name1, module1 in model.named_modules():
                        if isinstance(module1, QuantizationAwareModule):
                            if prev == name1:
                                if shared_threshold.get(name) is None:
                                    shared_threshold[name] = module1.activation_threshold
                                elif module1.activation_threshold > shared_threshold[name]:
                                    shared_threshold[name] = module1.activation_threshold
                for prev in prevs[name]:
                    for name1, module1 in model.named_modules():
                        if isinstance(module1, QuantizationAwareModule):
                            if prev == name1:
                                module1.activation_threshold = shared_threshold[name]

    # Get the thresholds after overrides
    thresholds = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizationAwareModule) or ("input_quantizer_hold" in name and isinstance(module, Eltwise)):
            thresholds[name] = module.activation_threshold

    print("Thresholds: ", thresholds)
    print("Previous layers: ", prevs)
    # Adjust bias and threshold values according to the previous layers,
    # and set the final scale value for output layers
    for name, module in model.named_modules():
        if isinstance(module, QuantizationAwareModule):
            if name in prevs:
                prev_threshold_set = False
                for name1, module1 in model.named_modules():
                    if isinstance(module1, QuantizationAwareModule) or "input_quantizer_hold" in name1:
                        if name1 in prevs[name]:
                            if name1 == "input_quantizer_hold":
                                print("Input quantizer hold")
                                print("Module: ", module)
                                print("Module1: ", module1)
                                print("Thresholds: ", thresholds[name1])
                            if not prev_threshold_set:
                                if module.op is not None and module.op.bias is not None:
                                    module.op.bias = nn.Parameter(module.op.bias /
                                                                  torch.exp2(thresholds[name1]))
                                module.activation_threshold = \
                                    nn.Parameter((module.activation_threshold - thresholds[name1]),
                                                 requires_grad=False)
                                if module.wide:
                                    module.final_scale = nn.Parameter(thresholds[name] -
                                                                      module.activation_threshold,
                                                                      requires_grad=False)
                                else:
                                    module.final_scale = nn.Parameter(thresholds[name],
                                                                      requires_grad=False)
                                prev_threshold_set = True
                            module1.final_scale = nn.Parameter(torch.tensor(0.),
                                                               requires_grad=False)


@torch.no_grad()
def stat_collect(train_loader, model, args):
    """Collect statistics for quantization aware training"""
    model.eval()
    #i = 0
    for inputs, _ in tqdm(train_loader):
        inputs = inputs.to(args.device)
        model(inputs)
        #i += 1
        #if i > 100:
        #    break

def pre_qat(model, train_loader, args, qat_policy):
    """
    Prepare the model for quantization aware training
    """

    init_hist(model)
    stat_collect(train_loader, model, args)
    init_threshold(model, qat_policy["outlier_removal_z_score"])
    release_hist(model)
    if args.fake_act_quant and args.lsq_weight_scale:
        set_device(dev.device, dev.simulate, dev.round_avg, args.fake_act_quant, args.per_channel, args.lsq_weight_scale)
        #print("Device configuration: ", dev)
    elif args.fake_act_quant and not args.lsq_weight_scale:
        set_device(dev.device, dev.simulate, dev.round_avg, args.fake_act_quant, args.per_channel, args.lsq_weight_scale)
        apply_scales(model)
    else:
        apply_scales(model)


def init_hist(model):
    """
    Place forward hooks to collect histograms of activations
    """
    for _, module in model.named_modules():
        if isinstance(module, (Eltwise, QuantizationAwareModule)):
            register_hist_hooks(module)


def release_hist(model):
    """
    Remove forward hooks after histogram collection
    """
    for _, module in model.named_modules():
        if isinstance(module, (Eltwise, QuantizationAwareModule)):
            release_hist_hooks(module)


def init_threshold(model, outlier_removal_z_score=8.0):
    """
    Calculate thresholds based on the collected histograms
    """
    for _, module in model.named_modules():
        if isinstance(module, (Eltwise, QuantizationAwareModule)):
            # If module defined but not called on forward, it won't have hist
            if hasattr(module, 'hist'):
                init_threshold_module(module, outlier_removal_z_score)
                calc_threshold(module)


def onnx_export_prep(m, simplify=False, remove_clamp=False):
    """
    Prepare model `m` for ONNX export. When `simplify` is True, remove several
    quantization related operators from the model graph.
    """
    def _onnx_export_prep(m):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if isinstance(target_attr, WeightScale):
                setattr(m, attr_str, WeightScaleONNX())
            elif isinstance(target_attr, OutputScale):
                setattr(m, attr_str, OutputScaleONNX())
            elif not simplify:
                if isinstance(target_attr, Quantize):
                    setattr(m, attr_str, QuantizeONNX(target_attr.num_bits))
                elif isinstance(target_attr, FloorQat):
                    setattr(m, attr_str, FloorQatONNX())
                elif isinstance(target_attr, RoundQat):
                    setattr(m, attr_str, RoundQatONNX())
                elif isinstance(target_attr, OutputShiftLimit):
                    setattr(m, attr_str, OutputShiftONNX())
                elif isinstance(target_attr, Scaler):
                    setattr(m, attr_str, ScalerONNX())
                elif isinstance(target_attr, Floor):
                    setattr(m, attr_str, FloorONNX())
                elif isinstance(target_attr, AvgPoolFloor):
                    setattr(m, attr_str, FloorONNX())
            elif isinstance(target_attr, (Quantize, Clamp, Round,
                                          AvgPoolFloor, Floor, FloorQat, RoundQat)):
                setattr(m, attr_str, Empty())
            elif isinstance(target_attr, OutputShiftLimit):
                if remove_clamp:
                    setattr(m, attr_str, ID3())
                else:
                    setattr(m, attr_str, OutputShiftONNX())
            elif isinstance(target_attr, Scaler):
                if remove_clamp:
                    setattr(m, attr_str, ID3())
                else:
                    setattr(m, attr_str, ScalerONNX())

    m.apply(_onnx_export_prep)


class bayer_filter:
    """
    Implement bayer filter to rgb images
    """
    def __call__(self, img):
        out = torch.zeros(1, img.shape[1], img.shape[2])

        out[0, 0::2, 1::2] = img[2, 0::2, 1::2]
        out[0, 0::2, 0::2] = img[1, 0::2, 0::2]
        out[0, 1::2, 1::2] = img[1, 1::2, 1::2]
        out[0, 1::2, 0::2] = img[0, 1::2, 0::2]

        return out
