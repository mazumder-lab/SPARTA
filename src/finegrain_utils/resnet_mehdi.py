# %%
"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from collections import OrderedDict

import torch.nn as nn

from finegrain_utils.utils_model_mehdi import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, with_mask=False, gamma=1.0, partially_trainable_bias=True):
        super(BasicBlock, self).__init__()
        self.with_mask = with_mask
        self.gamma = gamma
        self.partially_trainable_bias = partially_trainable_bias
        if self.with_mask:
            self.conv1 = Conv2d_partially_trainable(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                gamma=gamma,
                partially_trainable_bias=self.partially_trainable_bias,
            )
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.with_mask:
            self.conv2 = Conv2d_partially_trainable(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                gamma=gamma,
                partially_trainable_bias=self.partially_trainable_bias,
            )
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.with_mask:
                self.shortcut = nn.Sequential(
                    Conv2d_partially_trainable(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        gamma=gamma,
                        partially_trainable_bias=self.partially_trainable_bias,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, with_mask=False, gamma=1.0, partially_trainable_bias=True):
        super(Bottleneck, self).__init__()
        self.with_mask = with_mask
        self.gamma = gamma
        self.partially_trainable_bias = partially_trainable_bias
        if self.with_mask:
            self.conv1 = Conv2d_partially_trainable(
                in_planes,
                planes,
                kernel_size=1,
                bias=False,
                gamma=gamma,
                partially_trainable_bias=self.partially_trainable_bias,
            )
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if self.with_mask:
            self.conv2 = Conv2d_partially_trainable(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                gamma=gamma,
                partially_trainable_bias=self.partially_trainable_bias,
            )
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if self.with_mask:
            self.conv3 = Conv2d_partially_trainable(
                planes,
                self.expansion * planes,
                kernel_size=1,
                bias=False,
                gamma=gamma,
                partially_trainable_bias=self.partially_trainable_bias,
            )
        else:
            self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.with_mask:
                self.shortcut = nn.Sequential(
                    Conv2d_partially_trainable(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                        gamma=gamma,
                        partially_trainable_bias=self.partially_trainable_bias,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, with_mask=False, gamma=1.0, partially_trainable_bias=True):
        super(ResNet, self).__init__()
        self.with_mask = with_mask
        self.gamma = gamma
        self.partially_trainable_bias = partially_trainable_bias
        self.in_planes = 64

        if self.with_mask:
            self.conv1 = Conv2d_partially_trainable(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                gamma=gamma,
                partially_trainable_bias=self.partially_trainable_bias,
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            with_mask=self.with_mask,
            gamma=self.gamma,
            partially_trainable_bias=self.partially_trainable_bias,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            with_mask=self.with_mask,
            gamma=self.gamma,
            partially_trainable_bias=self.partially_trainable_bias,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            with_mask=self.with_mask,
            gamma=self.gamma,
            partially_trainable_bias=self.partially_trainable_bias,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            with_mask=self.with_mask,
            gamma=self.gamma,
            partially_trainable_bias=self.partially_trainable_bias,
        )

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block, planes, num_blocks, stride, with_mask=False, gamma=1.0, partially_trainable_bias=True
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, with_mask, gamma, partially_trainable_bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10, with_mask=False, gamma=1.0, partially_trainable_bias=True):
#         super(ResNet, self).__init__()
#         self.with_mask = with_mask
#         self.gamma = gamma
#         self.partially_trainable_bias = partially_trainable_bias
#         self.in_planes = 64

#         if self.with_mask:
#             self.conv1 = Conv2d_partially_trainable(
#                 3,
#                 64,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False,
#                 gamma=gamma,
#                 partially_trainable_bias=self.partially_trainable_bias,
#             )
#         else:
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(
#             block,
#             64,
#             num_blocks[0],
#             stride=1,
#             with_mask=self.with_mask,
#             gamma=self.gamma,
#             partially_trainable_bias=self.partially_trainable_bias,
#         )
#         self.layer2 = self._make_layer(
#             block,
#             128,
#             num_blocks[1],
#             stride=2,
#             with_mask=self.with_mask,
#             gamma=self.gamma,
#             partially_trainable_bias=self.partially_trainable_bias,
#         )
#         self.layer3 = self._make_layer(
#             block,
#             256,
#             num_blocks[2],
#             stride=2,
#             with_mask=self.with_mask,
#             gamma=self.gamma,
#             partially_trainable_bias=self.partially_trainable_bias,
#         )
#         self.layer4 = self._make_layer(
#             block,
#             512,
#             num_blocks[3],
#             stride=2,
#             with_mask=self.with_mask,
#             gamma=self.gamma,
#             partially_trainable_bias=self.partially_trainable_bias,
#         )

#         if self.with_mask:
#             self.linear = Linear_partially_trainable(
#                 512 * block.expansion, num_classes, gamma=gamma, partially_trainable_bias=self.partially_trainable_bias
#             )
#         else:
#             self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(
#         self, block, planes, num_blocks, stride, with_mask=False, gamma=1.0, partially_trainable_bias=True
#     ):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride, with_mask, gamma, partially_trainable_bias))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


def ResNet18_partially_trainable(num_classes=100, with_mask=False, gamma=1.0, partially_trainable_bias=True):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        with_mask=with_mask,
        gamma=gamma,
        partially_trainable_bias=partially_trainable_bias,
    )


def ResNet34_partially_trainable(with_mask=False, gamma=1.0, partially_trainable_bias=True):
    return ResNet(
        BasicBlock, [3, 4, 6, 3], with_mask=with_mask, gamma=gamma, partially_trainable_bias=partially_trainable_bias
    )


def ResNet50_partially_trainable(num_classes=100, with_mask=False, gamma=1.0, partially_trainable_bias=True):
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        with_mask=with_mask,
        gamma=gamma,
        partially_trainable_bias=partially_trainable_bias,
    )


def ResNet101_partially_trainable(with_mask=False, gamma=1.0, partially_trainable_bias=True):
    return ResNet(
        Bottleneck, [3, 4, 23, 3], with_mask=with_mask, gamma=gamma, partially_trainable_bias=partially_trainable_bias
    )


def ResNet152_partially_trainable(with_mask=False, gamma=1.0, partially_trainable_bias=True):
    return ResNet(
        Bottleneck, [3, 8, 36, 3], with_mask=with_mask, gamma=gamma, partially_trainable_bias=partially_trainable_bias
    )


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(len(net.get_parameter_groups("hsn")))
#     # print (list(net.named_parameters()))
#     print(y.size())


# %%

# %%

# %%

# # %%
# # Loading model with partially trainable weights (with_mask=True)
# # gamma isn't used for the moment (could be useful if we want to train the mask)

# # partially_trainable_bias=True if we want paritally trainable bias too
# net = ResNet18_partially_trainable(with_mask=True, gamma=1.0, partially_trainable_bias=True)


# # %%

# # %%

# # %%
# # Loading model with no partially trainable weights (with_mask=False): this is the regular model
# pruned_net = ResNet18_partially_trainable(with_mask=False, partially_trainable_bias=True)
# # Magnitude pruning per layer
# sparsity = 1.0
# d_params_pruned = dict(pruned_net.named_parameters())
# for name_param in d_params_pruned:
#     idx_weights = torch.argsort(d_params_pruned[name_param].flatten(), descending=False)
#     d_params_pruned[name_param].data.flatten()[idx_weights[: int(len(idx_weights) * sparsity)]] = 0
# # %%

# # %%
# # Starting from a pruned model to initialize the mask
# d_params = dict(net.named_parameters())
# for name_param in d_params:
#     if "mask" in name_param:
#         original_name_param = name_param.replace("mask_", "").replace("_trainable", "")
#         d_params[name_param][d_params_pruned[original_name_param] == 0] = 0
# # %% One gradient computation
# y = net(torch.randn(1, 3, 32, 32))
# loss = torch.nn.functional.cross_entropy(y, (y <= 0.0).float())
# loss.backward()
# # %% Checking that only a subset of the gradients are non zero:
# for name_param in d_params:
#     if "mask" in name_param:
#         original_name_param = name_param.replace("mask_", "").replace("_trainable", "")
#         print(
#             f"Sparsity in the gradients of {original_name_param}: {(d_params[original_name_param].grad==0).float().mean()}"
#         )

# # %% Checking that only a subset of the gradients are non zero:
# net_state_dict = dict(net.named_parameters())
# old_net_state_dict = dict(pruned_net.named_parameters())
# for original_name in net_state_dict:
#     if "init" in original_name:
#         name_mask = original_name.replace("init_", "mask_") + "_trainable"
#         name_weight = original_name.replace("init_", "") + "_trainable"
#         name = original_name.replace("_module.", "").replace("init_", "")
#         param = net_state_dict[original_name] + net_state_dict[name_mask] * net_state_dict[name_weight]
#         if name in old_net_state_dict:
#             print(f"Sparsity in {name}: {torch.mean((param - old_net_state_dict[name] == 0).float())}")

# # %%

# # %% Test same output with or without masking:
# net.eval()
# random_input = torch.randn(1, 3, 32, 32)
# use_mask_rec(net, True)
# y_mask = net(random_input)
# use_mask_rec(net, False)
# y_no_mask = net(random_input)
# print(y_mask - y_no_mask)
# # %%
# # %%
