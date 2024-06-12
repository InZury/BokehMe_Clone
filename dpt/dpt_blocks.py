import torch
import torch.nn as nn
import torch.nn.functional as func

from dpt_vit import make_pretrained_vit_base_resnet50_384


class ResidualConvUnitCustom(nn.Module):
    def __init__(self, features, activation, batch_norm):
        super(ResidualConvUnitCustom, self).__init__()

        self.batch_norm = batch_norm
        self.groups = 1
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=not self.batch_norm, groups=self.groups)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=not self.batch_norm, groups=self.groups)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(features)
            self.batch_norm2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, data):
        result = self.activation(data)
        result = self.conv1(result)

        if self.batch_norm:
            result = self.batch_norm1(result)

        result = self.activation(result)
        result = self.conv2(result)

        if self.batch_norm:
            result = self.batch_norm2(result)

        return self.skip_add(result, data)


class FeatureFusionBlockCustom(nn.Module):
    def __init__(self, features, activation, deconvolution=False, batch_norm=False, expand=False, align_corners=True):
        super(FeatureFusionBlockCustom, self).__init__()

        self.deconvolution = deconvolution
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features // 2 if self.expand else features

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)

        self.residual_conv_unit1 = ResidualConvUnitCustom(features, activation, batch_norm)
        self.residual_conv_unit2 = ResidualConvUnitCustom(features, activation, batch_norm)
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, *datas):
        result = datas[0]

        if len(datas) == 2:
            residual = self.residual_conv_unit1(datas[1])
            result = self.skip_add.add(result, residual)

        result = self.residual_conv_unit2(result)
        result = func.interpolate(result, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        result = self.out_conv(result)

        return result


def make_scratch(input_shape, output_shape, group=1, expand=False):
    scratch = nn.Module()

    if expand:
        output_shape1, output_shape2, output_shape3, output_shape4 = [output_shape * data for data in [1, 2, 4, 8]]
    else:
        output_shape1, output_shape2, output_shape3, output_shape4 = output_shape

    scratch.residual_layer1 = nn.Conv2d(
        input_shape[0], output_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
    scratch.residual_layer2 = nn.Conv2d(
        input_shape[1], output_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
    scratch.residual_layer3 = nn.Conv2d(
        input_shape[2], output_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
    scratch.residual_layer4 = nn.Conv2d(
        input_shape[3], output_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=group)

    return scratch


def make_encoder(
    backbone,
    features,
    use_pretrained,
    groups=1,
    expand=False,
    hooks=None,
    use_vit_only=False,
    use_readout='ignore',
    enable_attention_hooks=False
):
    if backbone == 'vit_base_resnet50_384':
        pretrained = make_pretrained_vit_base_resnet50_384(
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
            enable_attention_hooks=enable_attention_hooks
        )
        scratch = make_scratch([256, 512, 768, 768], features, group=groups, expand=expand)
    else:
        assert False, f"Backbone '{backbone}' is not implemented'"

    return pretrained, scratch
