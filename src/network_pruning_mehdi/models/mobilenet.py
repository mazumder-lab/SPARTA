#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from math import floor
import torch.nn as nn
from utils_model import Linear_with_z, Conv2d_with_z

__all__ = ['mobilenet', 'mobilenet_025', 'mobilenet_050', 'mobilenet_075']

class MobileNet(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8, with_z=True, gamma=1.0, prune_bias=True):
        super(MobileNet, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1, with_z=True, gamma=1.0, prune_bias=True):
            if with_z:
                return [
                    Conv2d_with_z(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False, gamma=gamma, prune_bias=prune_bias),
                    nn.BatchNorm2d(n_ofm),
                    nn.ReLU(inplace=True)
                ]
            else:
                return [
                    nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                    nn.BatchNorm2d(n_ofm),
                    nn.ReLU(inplace=True)
                ]

        def depthwise_conv(n_ifm, n_ofm, stride, with_z=True, gamma=1.0):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1, with_z=with_z, gamma=gamma, prune_bias=prune_bias)
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=2, padding=1, with_z=with_z, gamma=gamma, prune_bias=prune_bias)),
            depthwise_conv(self.channels[0], self.channels[1], 1, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[1], self.channels[2], 2, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[2], self.channels[2], 1, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[2], self.channels[3], 2, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[3], self.channels[3], 1, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[3], self.channels[4], 2, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[4], self.channels[4], 1, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[4], self.channels[4], 1, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[4], self.channels[4], 1, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[4], self.channels[4], 1, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[4], self.channels[4], 1, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[4], self.channels[5], 2, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            depthwise_conv(self.channels[5], self.channels[5], 1, with_z=with_z, gamma=gamma, prune_bias=prune_bias),
            nn.AvgPool2d(7),
        )
        if with_z:
            self.fc = Linear_with_z(self.channels[5], 1000, gamma=gamma, prune_bias=prune_bias)
        else:
            self.fc = nn.Linear(self.channels[5], 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        try:
            x = x[:,self.idx_keep_input]
        except:
            pass
        x = self.fc(x)
        return x


def mobilenet_025(gamma=1.0, with_z=True, prune_bias=True):
    return MobileNet(channel_multiplier=0.25, with_z=with_z, gamma=gamma, prune_bias=prune_bias)


def mobilenet_050(gamma=1.0, with_z=True, prune_bias=True):
    return MobileNet(channel_multiplier=0.5, with_z=with_z, gamma=gamma, prune_bias=prune_bias)


def mobilenet_075(gamma=1.0, with_z=True, prune_bias=True):
    return MobileNet(channel_multiplier=0.75, with_z=with_z, gamma=gamma, prune_bias=prune_bias)


def mobilenet(gamma=1.0, with_z=True, prune_bias=True, pretrained=False):
    return MobileNet(with_z=with_z, gamma=gamma, prune_bias=prune_bias)
