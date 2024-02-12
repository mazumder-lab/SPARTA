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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

# End of the imports for conv2d
def compute_z_from_tensor(input_tensor, gamma):
    condition_1 = input_tensor <= -gamma/2
    condition_2 = input_tensor >= gamma/2
    smooth_zs = (-2 /(gamma**3)) * (input_tensor**3) + (3/(2 * gamma)) * input_tensor + 0.5

    return torch.where(condition_1, torch.zeros_like(input_tensor), 
                        torch.where(condition_2, torch.ones_like(input_tensor), smooth_zs))

def compute_z_rec(module):
    name_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            compute_z_rec(child)
    elif not("relu" in name_module) and not("avgpool" in name_module) and not("norm" in name_module):
        try:
            module.compute_z()
        except:
            import ipdb;ipdb.set_trace()

def compute_n_z_rec(module):
    n_z = 0
    name_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            n_z += compute_n_z_rec(child)
    elif not("relu" in name_module) and not("avgpool" in name_module) and not("norm" in name_module):
        try:
            n_z += torch.sum(module.z>0, dtype=float).detach().item()
            if module.test_bias:
                n_z += torch.sum(module.z_2>0, dtype=float).detach().item()
        except:
            import ipdb;ipdb.set_trace()
    return n_z

def compute_n_z_close_to_1_rec(module, tol_z_1):
    n_z = 0
    name_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            n_z += compute_n_z_close_to_1_rec(child, tol_z_1)
    elif not("relu" in name_module) and not("avgpool" in name_module) and not("norm" in name_module):
        try:
            n_z+=torch.sum(module.z>=tol_z_1, dtype=float).detach().item()
            if module.test_bias:
                n_z+=torch.sum(module.z_2>=tol_z_1, dtype=float).detach().item()
        except:
            import ipdb;ipdb.set_trace()
    return n_z 

def reset_z_rec(module, tol_z_1, prop_reset):
    n_reset = 0
    input_tol = tol_z_1
    name_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            n_reset += reset_z_rec(child, tol_z_1, prop_reset)
    elif not("relu" in name_module) and not("avgpool" in name_module) and not("norm" in name_module):
        try:
            module.compute_z()
            old_z = module.z
            if module.test_bias:
                old_bias_z = module.z_2

            n_reset += module.reset_weight_z(input_tol, prop_reset)
            module.compute_z()
            new_z = module.z
            module.weight.data[new_z>0] *= (old_z/new_z)[new_z>0]
            if module.test_bias:
                new_bias_z = module.z_2
                module.bias.data[new_bias_z>0] *= (old_bias_z/new_bias_z)[new_bias_z>0]
        except:
            import ipdb;ipdb.set_trace()
    return n_reset

def compute_losses_rec(module, device, entropy_reg, selection_reg, l2_reg):
    entropy_loss = torch.tensor(0.0).to(device)
    selection_loss = torch.tensor(0.0).to(device)
    l2_loss = torch.tensor(0.0).to(device)
    name_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            entropy_loss_child, selection_loss_child, l2_loss_child = compute_losses_rec(child, device, entropy_reg, selection_reg, l2_reg)
            entropy_loss += entropy_loss_child
            selection_loss += selection_loss_child
            l2_loss += l2_loss_child
    elif not("relu" in name_module) and not("avgpool" in name_module) and not("norm" in name_module):
        try:
            if entropy_reg!=0:
                entropy_loss += -entropy_reg*(torch.sum(module.z*torch.log(module.z+1e-6) + (1-module.z)*torch.log(1-module.z+1e-6)))
                if module.test_bias:
                    entropy_loss += -entropy_reg*(torch.sum(module.z_2*torch.log(module.z_2+1e-6) + (1-module.z_2)*torch.log(1-module.z_2+1e-6)))
            if selection_reg !=0:
                selection_loss += selection_reg*torch.sum(module.z)
                if module.test_bias:
                    selection_loss += selection_reg*torch.sum(module.z_2)
            if l2_reg !=0:
                l2_loss += l2_reg*torch.sum(module.weight**2)
                if module.test_bias:
                    l2_loss += l2_reg*torch.sum(module.bias**2)
            return entropy_loss, selection_loss, l2_loss
        except:
            import ipdb;ipdb.set_trace()
    return entropy_loss, selection_loss, l2_loss

def prune_models_rec(module):
    test_pruned = False
    name_module = module.__str__().lower()
    d_children = dict(module.named_children())
    test_children = len(d_children)>0
    if test_children:
        for name_child in d_children:
            child = d_children[name_child]
            test_pruned_child = prune_models_rec(child)
            test_pruned = (test_pruned or test_pruned_child)
    elif not("relu" in name_module) and not("avgpool" in name_module) and not("norm" in name_module):
        try:
            if torch.min(module.z) == 0:
                module.weight_z.data[module.z==0] = -module.gamma
                module.weight.data[module.z==0] = 0
                test_pruned = True
            if module.test_bias:
                if torch.min(module.z_2) == 0:
                    module.bias_z.data[module.z_2==0] = -module.gamma
                    module.bias.data[module.z_2==0] = 0
                    test_pruned = True
        except:
            import ipdb;ipdb.set_trace()
    return test_pruned

class Linear_with_z(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gamma: float = 1.0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma=gamma
        self.is_sparse = False
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_z = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.test_bias = bias

        if self.test_bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_z = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_z', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #self.weight_z.data.uniform_(-self.gamma/100, self.gamma/100)
        self.weight_z.data.uniform_(self.gamma, self.gamma)

        if self.test_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            #self.bias_z.data.uniform_(-self.gamma/100, self.gamma/100)
            self.bias_z.data.uniform_(self.gamma, self.gamma)

    def reset_weight_z(self, tol_z_1, prop_reset=1.0) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        n_weights = np.prod(self.weight.shape)
        n_to_reset = int(prop_reset*n_weights)
        n_reset = 0
        with torch.no_grad():
            idx_weight = torch.argsort(torch.abs(self.weight).view(-1))
            idx_weight = idx_weight[self.z.view(-1)[idx_weight]>=tol_z_1][:n_to_reset]
            n_reset+=len(idx_weight)
            # print(self.weight.view(-1).data[idx_weight])
            self.weight_z.view(-1).data[idx_weight] = self.weight_z.view(-1).data[idx_weight].uniform_(-self.gamma/100, self.gamma/100)

            if self.test_bias:
                idx_bias = torch.argsort(torch.abs(self.bias).view(-1))
                idx_bias = idx_bias[self.z_2.view(-1)[idx_bias]>=tol_z_1][:n_to_reset]
                n_reset+=len(idx_weight)
                self.bias_z.view(-1).data[idx_bias] = self.bias_z.view(-1).data[idx_bias].uniform_(-self.gamma/100, self.gamma/100)
            
        return n_reset

    def compute_z(self) -> None:
        # condition_1 = self.weight_z <= -self.gamma/2
        # condition_2 = self.weight_z >= self.gamma/2
        # smooth_zs = (-2 /(self.gamma**3)) * (self.weight_z**3) + (3/(2 * self.gamma)) * self.weight_z + 0.5

        # self.z = torch.where(condition_1, torch.zeros_like(self.weight_z), 
        #                     torch.where(condition_2, torch.ones_like(self.weight_z), smooth_zs))

        self.z = compute_z_from_tensor(self.weight_z, self.gamma)

        if self.test_bias:
            # condition_1 = self.bias_z <= -self.gamma/2
            # condition_2 = self.bias_z >= self.gamma/2
            # smooth_zs = (-2 /(self.gamma**3)) * (self.bias_z**3) + (3/(2 * self.gamma)) * self.bias_z + 0.5

            # self.z_2 = torch.where(condition_1, torch.zeros_like(self.bias_z), 
            #                     torch.where(condition_2, torch.ones_like(self.bias_z), smooth_zs))
            self.z_2 = compute_z_from_tensor(self.bias_z, self.gamma)
            self.bias_final = torch.mul(self.bias,self.z_2)
        else:
            self.bias_final = self.bias

    def forward(self, input: Tensor) -> Tensor:
        self.compute_z()
        return F.linear(input, self.weight*self.z, self.bias_final)

    def sparsify(self) -> None:
        sparse_weight = self.weight.data.to_sparse()
        self.sparse_weight_values = Parameter(sparse_weight.values())
        self.sparse_weight_indices = Parameter(sparse_weight.indices(), requires_grad=False)
        self.sparse_weight_size = sparse_weight.size()
        del self.weight
        sparse_weight_z = self.weight_z.data.to_sparse()
        self.sparse_weight_z_values = Parameter(sparse_weight_z.values())
        del self.weight_z
        if self.test_bias:
            sparse_bias = self.bias.data.to_sparse()
            self.sparse_bias_values = Parameter(sparse_bias.values())
            self.sparse_bias_indices = Parameter(sparse_bias.indices(), requires_grad=False)
            self.sparse_bias_size = sparse_bias.size()
            del self.bias
            sparse_bias_z = self.bias_z.data.to_sparse()
            self.sparse_bias_z_values = Parameter(sparse_bias_z.values())
            del self.bias_z
        self.is_sparse = True

    def freeze_z(self) -> None:
        self.weight_z.requires_grad = False
        if self.test_bias:
            self.bias_z.requires_grad = False

    def unfreeze_z(self) -> None:
        self.weight_z.requires_grad = True
        if self.test_bias:
            self.bias_z.requires_grad = True

    def freeze_weight(self) -> None:
        self.weight.requires_grad = False
        if self.test_bias:
            self.bias.requires_grad = False

    def unfreeze_weight(self) -> None:
        self.weight.requires_grad = True
        if self.test_bias:
            self.bias.requires_grad = True

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def conv3x3(in_planes, out_planes, stride=1, gamma=1.0, with_z=True):
    """3x3 convolution with padding"""
    if with_z:
        return Conv2d_with_z(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, gamma=gamma)
    else:
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class _ConvNd_with_z(torch.nn.Module):

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
                 gamma:float = 1.0) -> None:
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
            self.weight_z = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
            self.weight_z = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
            self.bias_z = Parameter(torch.empty(out_channels, **factory_kwargs))
            self.test_bias = True
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_z', None)
            self.test_bias = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #self.weight_z.data.uniform_(-self.gamma/100, self.gamma/100)
        self.weight_z.data.uniform_(self.gamma, self.gamma)

        if self.test_bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
                self.bias_z.data.uniform_(self.gamma, self.gamma)

    def reset_weight_z(self, tol_z_1, prop_reset=1.0) -> None:
        n_weights = np.prod(self.weight.shape)
        n_to_reset = int(prop_reset*n_weights)
        n_reset = 0
        with torch.no_grad():
            idx_weight = torch.argsort(torch.abs(self.weight).view(-1))
            idx_weight = idx_weight[self.z.view(-1)[idx_weight]>=tol_z_1][:n_to_reset]
            n_reset+=len(idx_weight)
            self.weight_z.view(-1).data[idx_weight] = self.weight_z.view(-1).data[idx_weight].uniform_(-self.gamma/100, self.gamma/100)

            if self.test_bias:
                idx_bias = torch.argsort(torch.abs(self.bias).view(-1))
                idx_bias = idx_bias[self.z_2.view(-1)[idx_bias]>=tol_z_1][:n_to_reset]
                n_reset+=len(idx_weight)
                self.bias_z.view(-1).data[idx_bias] = self.bias_z.view(-1).data[idx_bias].uniform_(-self.gamma/100, self.gamma/100)
            
        return n_reset

    def compute_z(self) -> None:
        self.z = compute_z_from_tensor(self.weight_z, self.gamma)

        if self.test_bias:
            self.z_2 = compute_z_from_tensor(self.bias_z, self.gamma)
            self.bias_final = torch.mul(self.bias,self.z_2)
        else:
            self.bias_final = self.bias
  
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
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class Conv2d_with_z(_ConvNd_with_z):
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
        gamma:float=1.0
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, gamma=gamma, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        self.compute_z()
        return self._conv_forward(input, self.weight*self.z, self.bias_final)

class model_wrapper():
    def __init__(self, model, seed, entropy_reg, selection_reg, l2_reg, device, dense_to_sparse, test_different_lr, steps_per_epoch, val_second_lr, momentum, weight_decay, n_params_original_z, tol_z_1) -> None:
        self.model = model
        self.seed = seed
        self.entropy_reg = entropy_reg
        self.selection_reg = selection_reg
        self.l2_reg = l2_reg
        self.device = device
        self.dense_to_sparse = dense_to_sparse
        self.test_different_lr = test_different_lr
        self.steps_per_epoch = steps_per_epoch
        self.val_second_lr = val_second_lr 
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_params_original_z = n_params_original_z
        self.tol_z_1 = tol_z_1

    def get_losses(self) -> float:
        return compute_losses_rec(self.model, self.device, self.entropy_reg, self.selection_reg, self.l2_reg)
        # entropy_loss = torch.tensor(0.0).to(self.device)
        # selection_loss = torch.tensor(0.0).to(self.device)
        # l2_loss = torch.tensor(0.0).to(self.device)
        # d_children = dict(self.model.named_children())
        # for name_child in d_children:
        #     if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #         child = d_children[name_child]
        #         if self.entropy_reg!=0:
        #             entropy_loss += -self.entropy_reg*(torch.sum(child.z*torch.log(child.z+1e-6) + (1-child.z)*torch.log(1-child.z+1e-6)))
        #             if child.bias != None:
        #                 entropy_loss += -self.entropy_reg*(torch.sum(child.z_2*torch.log(child.z_2+1e-6) + (1-child.z_2)*torch.log(1-child.z_2+1e-6)))
        #         if self.selection_reg !=0:
        #             selection_loss += self.selection_reg*torch.sum(child.z)
        #             if child.bias != None:
        #                 selection_loss += self.selection_reg*torch.sum(child.z_2)
        #         if self.l2_reg !=0:
        #             l2_loss += self.l2_reg*torch.sum(child.weight**2)
        #             if child.bias != None:
        #                 l2_loss += self.l2_reg*torch.sum(child.bias**2)
        # return entropy_loss, selection_loss, l2_loss

    def get_n_z(self) -> float:
        return compute_n_z_rec(self.model)
        # n_z = 0
        # d_children = dict(self.model.named_children())
        # for name_child in d_children:
        #     if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #         child = d_children[name_child]
        #         n_z += torch.sum(child.z>0, dtype=float)
        # return n_z.detach().item()

    def get_n_z_close_to_1(self) -> float:
        return compute_n_z_close_to_1_rec(self.model, self.tol_z_1)
        # n_z = 0
        # d_children = dict(self.model.named_children())
        # for name_child in d_children:
        #     if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #         child = d_children[name_child]
        #         n_z += torch.sum(child.z>=self.tol_z_1, dtype=float)
        # return n_z.detach().item()

    def compute_z(self) -> float:
        compute_z_rec(self.model)
        # d_children = dict(self.model.named_children())
        # for name_child in d_children:
        #     child = d_children[name_child]
        #     test_children = False
        #     try:
        #         child.named_children()
        #         test_children = True
        #     except:
        #         pass
        #     if test_children:
        #         child.compute_z()
        #     # if "layer" in name_child:
        #     #     d_children_layer = dict(child.named_children())
        #     #     for name_child_layer in d_children_layer:
        #     #         child_layer = d_children_layer[name_child_layer]
        #     #         if not("relu" in name_child_layer) and not("avgpool" in name_child_layer) and not("bn" in name_child_layer):
        #     #             child_layer.compute_z()
        #     elif not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #         try:
        #             child.compute_z()
        #         except:
        #             import ipdb;ipdb.set_trace()

    def reset_z(self, prop_reset = 0.5) -> float:
        n_reset = reset_z_rec(self.model, self.tol_z_1, prop_reset=prop_reset)
        # # if initialization:
        # #     input_tol = 0.0
        # # else:
        # input_tol = self.tol_z_1
        # n_reset = 0
        # with torch.no_grad():
        #     d_children = dict(self.model.named_children())
        #     for name_child in d_children:
        #         if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #             child = d_children[name_child]
        #             # if initialization:
        #             #     old_z = torch.ones_like(child.weight_z)
        #             #     if child.test_bias:
        #             #         old_bias_z = torch.ones_like(child.bias_z)
        #             # else:
        #             child.compute_z()
        #             old_z = child.z
        #             if child.test_bias:
        #                 old_bias_z = child.z_2

        #             n_reset += child.reset_weight_z(input_tol, prop_reset)
        #             child.compute_z()
        #             new_z = child.z
        #             child.weight.data[new_z>0] *= (old_z/new_z)[new_z>0]
        #             if child.test_bias:
        #                 new_bias_z = child.z_2
        #                 child.bias.data[new_bias_z>0] *= (old_bias_z/new_bias_z)[new_bias_z>0]
        return n_reset

    def multiply_weight(self) -> None:
        parameters = self.model.named_parameters()
        with torch.no_grad():
            for parameter in parameters:
                if not("_z" in parameter[0]) and not("indices" in parameter[0]):
                    parameter[1].data = 2*parameter[1].data

    def multiply_weight_copy(self) -> None:
        parameters = self.model.named_parameters()
        with torch.no_grad():
            for parameter in parameters:
                if not("_z" in parameter[0]) and not("indices" in parameter[0]):
                    parameter[1].data = 2*parameter[1].data

    def sparsify(self) -> None:
        d_children = dict(self.model.named_children())
        for name_child in d_children:
            if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
                child = d_children[name_child]
                child.sparsify()

    def phase_training_z(self) -> None:
        d_children = dict(self.model.named_children())
        for name_child in d_children:
            if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
                child = d_children[name_child]
                #child.unfreeze_z()
                child.freeze_weight()

    def phase_training_weight(self) -> None:
        d_children = dict(self.model.named_children())
        for name_child in d_children:
            if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
                child = d_children[name_child]
                #child.freeze_z()
                child.unfreeze_weight()

    def prune_models(self, optimizer):
        test_pruned = prune_models_rec(self.model)
        return optimizer, test_pruned
        # test_pruned = False
        # if self.dense_to_sparse:
        #     with torch.no_grad():
        #         d_children = dict(self.model.named_children())
        #         for name_child in d_children:
        #             if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #                 child = d_children[name_child]
        #                 if len(child.z)>0 and torch.min(child.z) == 0:
        #                     optimizer_name = optimizer.__class__.__name__
        #                     idx_keep = torch.where(child.z!=0)[0]
        #                     copy_grad_weight = copy.deepcopy(child.sparse_weight_values.grad[idx_keep])
        #                     copy_grad_weight_z = copy.deepcopy(child.sparse_weight_z_values.grad[idx_keep])
        #                     sparse_weight_values_temp = Parameter(child.sparse_weight_values[idx_keep])
        #                     sparse_weight_z_values_temp = Parameter(child.sparse_weight_z_values[idx_keep])
        #                     if optimizer_name == "SGD":
        #                         optimizer.state[sparse_weight_values_temp]["momentum_buffer"] = copy.deepcopy(optimizer.state[child.sparse_weight_values].pop("momentum_buffer")[idx_keep])
        #                         optimizer.state[sparse_weight_z_values_temp]["momentum_buffer"] = copy.deepcopy(optimizer.state[child.sparse_weight_z_values].pop("momentum_buffer")[idx_keep])
        #                     elif optimizer_name == "Adam":
        #                         optimizer.state[sparse_weight_values_temp]["exp_avg"] = copy.deepcopy(optimizer.state[child.sparse_weight_values].pop("exp_avg")[idx_keep])
        #                         optimizer.state[sparse_weight_z_values_temp]["exp_avg"] = copy.deepcopy(optimizer.state[child.sparse_weight_z_values].pop("exp_avg")[idx_keep])
        #                         optimizer.state[sparse_weight_values_temp]["exp_avg_sq"] = copy.deepcopy(optimizer.state[child.sparse_weight_values].pop("exp_avg_sq")[idx_keep])
        #                         optimizer.state[sparse_weight_z_values_temp]["exp_avg_sq"] = copy.deepcopy(optimizer.state[child.sparse_weight_z_values].pop("exp_avg_sq")[idx_keep])
        #                     child.sparse_weight_values = sparse_weight_values_temp
        #                     child.sparse_weight_z_values = sparse_weight_z_values_temp
        #                     child.sparse_weight_values.grad = copy_grad_weight
        #                     child.sparse_weight_z_values.grad = copy_grad_weight_z
        #                     child.sparse_weight_indices = Parameter(child.sparse_weight_indices[:,idx_keep], requires_grad=False)
                            
        #                     if child.bias:
        #                         # NEED TO IMPLEMENT PRUNING FOR BIAS
        #                         import ipdb;ipdb.set_trace()
        #                     test_pruned = True

        #     if test_pruned:
        #         # self.update_indice_param()
        #         optimizer_name = optimizer.__class__.__name__
        #         copy_optimizer = initialize_optimizer(self.test_different_lr, self.model, optimizer_name, self.steps_per_epoch, optimizer.defaults["lr"], self.val_second_lr, self.momentum, self.weight_decay)
        #         try:
        #             copy_optimizer._step_count = optimizer._step_count
        #         except:
        #             pass
        #         for idx_param in range(len(optimizer.param_groups)):
        #             for key in optimizer.param_groups[idx_param]:
        #                 if key!="params":
        #                     copy_optimizer.param_groups[idx_param][key] = optimizer.param_groups[idx_param][key]
        #             list_params_old = list(optimizer.param_groups[idx_param]["params"])
        #             list_params_new = list(copy_optimizer.param_groups[idx_param]["params"])
        #             for i in range(len(list_params_new)):
        #                 group_param_old = list_params_old[i]
        #                 group_param_new = list_params_new[i]
        #                 copy_optimizer.state[group_param_new] = copy.deepcopy(optimizer.state[group_param_old])
        #     else:
        #         copy_optimizer = optimizer
        # else:
        #     with torch.no_grad():
        #         d_children = dict(self.model.named_children())
        #         for name_child in d_children:
        #             if not("relu" in name_child) and not("avgpool" in name_child) and not("bn" in name_child):
        #                 child = d_children[name_child]
        #                 if torch.min(child.z) == 0:
        #                     child.weight_z.data[child.z==0] = -child.gamma
        #                     test_pruned = True
        #     copy_optimizer = optimizer
        # return copy_optimizer, test_pruned

# %%
