# encoding utf-8

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as func


class Space2Depth(nn.Module):
    def __init__(self, down_factor):
        super(Space2Depth, self).__init__()
        self.down_factor = down_factor

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = func.unfold(x, kernel_size=self.down_factor, stride=self.down_factor)
        return unfolded_x.view(n, c * self.down_factor ** 2, h // self.down_factor, w // self.down_factor)


def set_module(in_channels, out_channels, kernel_size, stride, padding, batch_norm, activation):
    module = nn.Sequential()

    # module.add_module('pad', nn.ReflectionPad2d(padding))
    module.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    module.add_module('bn', nn.BatchNorm2d(out_channels)) if batch_norm else None
    module.add_module('activation', activation) if activation else None

    return module


class BlockStack(nn.Module):
    # connect_mode: refer to "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks"
    def __init__(self, channels, num_block, share_weight, connect_mode, batch_norm, activation):
        super(BlockStack, self).__init__()

        self.num_block = num_block
        self.connect_mode = connect_mode
        self.blocks = nn.ModuleList()

        if share_weight is True:
            block = nn.Sequential(
                set_module(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3, stride=1, padding=1,
                    batch_norm=batch_norm,
                    activation=activation
                ),
                set_module(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3, stride=1, padding=1,
                    batch_norm=batch_norm,
                    activation=activation
                )
            )
            for i in range(num_block):
                self.blocks.append(block)
            # for end
        else:
            for i in range(num_block):
                block = nn.Sequential(
                    set_module(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3, stride=1, padding=1,
                        batch_norm=batch_norm, activation=activation
                    ),
                    set_module(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3, stride=1, padding=1,
                        batch_norm=batch_norm, activation=activation
                    )
                )
                self.blocks.append(block)
            # for end
        # if end

    def forward(self, x):
        if self.connect_mode == 'no':
            for i in range(self.num_block):
                x = self.blocks[i](x)
        elif self.connect_mode == 'distinct_source':
            for i in range(self.num_block):
                x = self.blocks[i](x) + x
        elif self.connect_mode == 'shared_source':
            default_x = x
            for i in range(self.num_block):
                x = self.blocks[i](x) + default_x
        else:
            raise Exception("'connect_mode' error!")

        return x


class ARNet(nn.Module):  # Adaptive Rendering Network
    def __init__(self, shuffle_rate=2, in_channels=5, out_channels=4, middle_channels=128, num_block=3,
                 share_weight=False, connect_mode='distinct_source', batch_norm=False, activation='elu'):
        super(ARNet, self).__init__()

        self.shuffle_rate = shuffle_rate
        self.connect_mode = connect_mode

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation = nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            activation = nn.ELU(inplace=True)
        else:
            raise Exception("'activation' error!")
        # if end

        self.down_sample = Space2Depth(shuffle_rate)
        self.conv0 = set_module(
            in_channels=(in_channels - 1) * shuffle_rate ** 2 + 1,
            out_channels=middle_channels,
            kernel_size=3, stride=1, padding=1,
            batch_norm=batch_norm, activation=activation
        )
        self.block_stack = BlockStack(
            channels=middle_channels,
            num_block=num_block, share_weight=share_weight, connect_mode=connect_mode,
            batch_norm=batch_norm, activation=activation
        )
        self.conv1 = set_module(
            in_channels=middle_channels,
            out_channels=out_channels * shuffle_rate ** 2,
            kernel_size=3, stride=1, padding=1,
            batch_norm=batch_norm, activation=activation
        )
        self.up_sample = nn.PixelShuffle(shuffle_rate)

    def forward(self, image, refocus, gamma):
        _, _, h, w = image.shape
        resize_h = int(h // self.shuffle_rate * self.shuffle_rate)
        resize_w = int(w // self.shuffle_rate * self.shuffle_rate)
        x0 = torch.cat((image, refocus), dim=1)
        x1 = func.interpolate(x0, size=(resize_h, resize_w), mode='bilinear', align_corners=True)
        x2 = self.down_sample(x1)
        gamma = torch.ones_like(x2[:, :1]) * gamma
        x3 = torch.cat((x2, gamma), dim=1)
        x4 = self.conv0(x3)
        x5 = self.block_stack(x4)
        x6 = self.conv1(x5)
        x7 = self.up_sample(x6)
        x8 = func.interpolate(x7, size=(h, w), mode='bilinear', align_corners=True)

        bokeh = x8[:, :-1]
        mask = torch.sigmoid(x8[:, -1:])

        return bokeh, mask


class IUNet(nn.Module):  # iterative up-sampling network
    def __init__(self, shuffle_rate=2, in_channels=8, out_channels=3, middle_channels=64, num_block=3,
                 share_weight=False, connect_mode='distinct_source', batch_norm=False, activation='elu'):
        super(IUNet, self).__init__()

        self.shuffle_rate = shuffle_rate
        self.connect_mode = connect_mode

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation = nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            activation = nn.ELU(inplace=True)
        else:
            raise Exception("'activation' error!")
        # if end

        self.down_sample = Space2Depth(shuffle_rate)
        self.conv0 = set_module(
            in_channels=(in_channels - 4) * shuffle_rate ** 2 + 4,
            out_channels=middle_channels,
            kernel_size=3, stride=1, padding=1,
            batch_norm=batch_norm, activation=activation
        )
        self.block_stack = BlockStack(
            channels=middle_channels,
            num_block=num_block, share_weight=share_weight, connect_mode=connect_mode,
            batch_norm=batch_norm, activation=activation
        )
        self.conv1 = set_module(
            in_channels=middle_channels,
            out_channels=out_channels * shuffle_rate ** 2,
            kernel_size=3, stride=1, padding=1,
            batch_norm=False, activation=None
        )
        self.up_sample = nn.PixelShuffle(shuffle_rate)

    def forward(self, image, refocus, bokeh_coarse, gamma):
        _, _, h, w = image.shape
        resize_h = int(h // self.shuffle_rate * self.shuffle_rate)
        resize_w = int(w // self.shuffle_rate * self.shuffle_rate)
        x0 = torch.cat((image, refocus), dim=1)
        x1 = func.interpolate(x0, size=(resize_h, resize_w), mode='bilinear', align_corners=True)
        x2 = self.down_sample(x1)

        if bokeh_coarse.shape[2] != x2.shape[2] or bokeh_coarse.shape[3] != x2.shape[3]:
            bokeh_coarse = func.interpolate(bokeh_coarse, size=(x2.shape[2], x2.shape[3]),
                                            mode='bilinear', align_corners=True)
        # if end

        gamma = torch.ones_like(x2[:, :1]) * gamma
        x3 = torch.cat((x2, bokeh_coarse, gamma), dim=1)
        x4 = self.conv0(x3)
        x5 = self.block_stack(x4)
        x6 = self.conv1(x5)
        x7 = self.up_sample(x6)
        bokeh_refine = func.interpolate(x7, size=(h, w), mode='bilinear', align_corners=True)

        return bokeh_refine
