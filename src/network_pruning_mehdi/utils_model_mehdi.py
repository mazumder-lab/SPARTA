#%%
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
#%%
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import math
import copy
from utils_optimizer import *

# Imports for conv2d
from typing import TypeVar, Union, Tuple, Optional, List
from itertools import repeat
import collections

import time

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]
_pair = _ntuple(2, "_pair")

def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

def use_mask_rec(module, use_mask):
    str_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            use_mask_rec(child, use_mask)
    elif "mask" in str_module:
        module.use_mask = use_mask

class Linear_partially_trainable(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gamma: float = 1.0,
                 device=None, dtype=None, partially_trainable_bias=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma=gamma
        self.is_sparse = False
        self.w_old = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.mask_weight_trainable = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        self.test_bias = bias
        self.partially_trainable_bias=partially_trainable_bias
        self.use_mask=True

        if self.test_bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            if self.partially_trainable_bias:
                self.mask_bias_trainable = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
            else:
                self.register_parameter('mask_bias_trainable', None)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mask_bias_trainable', None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        try:
            del self.bias_final
        except:
            pass
        if self.test_bias and self.partially_trainable_bias:
            if self.use_mask:
                bias_old = self.bias.data.detach()
                self.bias_final = self.mask_bias_trainable*self.bias + (1-self.mask_bias_trainable)*bias_old
            else:
                self.bias_final = self.bias
        else:
            self.bias_final = self.bias

        if self.use_mask:
            #w_old = self.weight.data.detach()
            output_linear = F.linear(input, self.weight*self.mask_weight_trainable + self.w_old*(1-self.mask_weight_trainable), self.bias_final)
        else:
            output_linear = F.linear(input, self.weight, self.bias_final)
        return output_linear

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.mask_weight_trainable.data.uniform_(1.0, 1.0)

        if self.test_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            if self.partially_trainable_bias:
                self.mask_bias_trainable.data.uniform_(1.0, 1.0)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, gamma={}, p_t_bias={}, use_mask={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.gamma, self.partially_trainable_bias, self.use_mask
        )

    def __deepcopy__(self, memo):
        out = Linear_partially_trainable(
                in_features = self.in_features,
                out_features = self.out_features,
                bias = self.test_bias,
                gamma = self.gamma,
                partially_trainable_bias= self.partially_trainable_bias)
    
        out.in_features=self.in_features
        out.out_features=self.out_features
        out.is_sparse=self.is_sparse
        out.weight=copy.deepcopy(self.weight, memo)
        out.mask_weight_trainable=copy.deepcopy(self.mask_weight_trainable, memo)
        out.test_bias=self.test_bias
        out.partially_trainable_bias=self.partially_trainable_bias
        out.use_mask=self.use_mask
        out.gamma = self.gamma,

        if self.test_bias:
            out.bias = copy.deepcopy(self.bias, memo)
            if self.partially_trainable_bias:
                out.mask_bias_trainable = copy.deepcopy(self.mask_bias_trainable, memo)
        return out

def conv3x3(in_planes, out_planes, stride=1, gamma=1.0, with_z=True, partially_trainable_bias=True):
    """3x3 convolution with padding"""
    if with_z:
        return Conv2d_partially_trainable(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, gamma=gamma, partially_trainable_bias=partially_trainable_bias)
    else:
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class _ConvNd_partially_trainable(torch.nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None,
                 gamma:float = 1.0,
                 partially_trainable_bias=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.partially_trainable_bias=partially_trainable_bias
        self.use_mask=True
        self.gamma=gamma
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
            self.mask_weight_trainable = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs), requires_grad=False)
            self.w_old = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
            self.mask_weight_trainable = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs), requires_grad=False)
            self.w_old = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
            if self.partially_trainable_bias:
                self.mask_bias_trainable = Parameter(torch.empty(out_channels, **factory_kwargs), requires_grad=False)
            else:
                self.register_parameter('mask_bias_trainable', None)
            self.test_bias = True
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mask_bias_trainable', None)
            self.test_bias = False

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.mask_weight_trainable.data.uniform_(1.0, 1.0)

        if self.test_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
                if self.partially_trainable_bias:
                    self.mask_bias_trainable.data.uniform_(1.0, 1.0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', gamma={gamma}'
        s += ', p_t_bias={partially_trainable_bias}'
        s += ', use_mask={use_mask}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class Conv2d_partially_trainable(_ConvNd_partially_trainable):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        gamma:float=1.0,
        partially_trainable_bias=True
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, gamma=gamma, partially_trainable_bias=partially_trainable_bias, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        self.bias_final = None
        #w_old = self.weight.data.detach()

        if self.test_bias and self.partially_trainable_bias:
            if self.use_mask:
                bias_old = self.bias.data.detach()
                self.bias_final = self.mask_bias_trainable*self.bias + (1-self.mask_bias_trainable)*bias_old
            else:
                self.bias_final = self.bias
        else:
            self.bias_final = self.bias

        if self.use_mask:
            return self._conv_forward(input, self.weight*self.mask_weight_trainable + self.w_old*(1-self.mask_weight_trainable),self.bias_final)
        else:
            return self._conv_forward(input, self.weight, self.bias_final)

