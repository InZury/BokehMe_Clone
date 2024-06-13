import torch
import torch.nn as nn
from .dpt_blocks import (FeatureFusionBlockCustom, make_encoder)
from .dpt_vit import forward_vit


class BaseModel(nn.Module):
    def load(self, path):
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters['model']

        self.load_state_dict(parameters)


class DPT(BaseModel):
    def __init__(self, head, features=256, backbone='vit_base_resnet50_384', readout='projection',
                 channels_last=False, batch_norm=False, enable_attention_hooks=False):
        super(DPT, self).__init__()

        self.channels_last = channels_last
        self.pretrained, self.scratch = make_encoder(
            backbone, features, use_pretrained=False, groups=1, expand=False, hooks=[0, 1, 8, 11],
            use_readout=readout, enable_attention_hooks=enable_attention_hooks)

        self.scratch.refinenet1 = make_fusion_block(features, batch_norm)
        self.scratch.refinenet2 = make_fusion_block(features, batch_norm)
        self.scratch.refinenet3 = make_fusion_block(features, batch_norm)
        self.scratch.refinenet4 = make_fusion_block(features, batch_norm)

        self.scratch.output_conv = head

    def forward(self, data):
        if self.channels_last:
            data.contiguous(memory_format=torch.channels_last)

        layer1, layer2, layer3, layer4 = forward_vit(self.pretrained, data)

        residual_layer1 = self.scratch.layer1_rn(layer1)
        residual_layer2 = self.scratch.layer2_rn(layer2)
        residual_layer3 = self.scratch.layer3_rn(layer3)
        residual_layer4 = self.scratch.layer4_rn(layer4)

        path4 = self.scratch.refinenet4(residual_layer4)
        path3 = self.scratch.refinenet3(path4, residual_layer3)
        path2 = self.scratch.refinenet2(path3, residual_layer2)
        path1 = self.scratch.refinenet1(path2, residual_layer1)

        result = self.scratch.output_conv(path1)

        return result


class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs):

        features = kwargs['features'] if 'features' in kwargs else 256
        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True) if non_negative else nn.Identity(),
            nn.Identity()
        )

        super(DPTDepthModel, self).__init__(head, **kwargs)

        self.scale = scale
        self.shift = shift
        self.invert = invert

        if path is not None:
            self.load(path)

    def forward(self, data):
        invert_depth = super(DPTDepthModel, self).forward(data).squeeze(dim=1)

        if self.invert:
            depth = self.scale * invert_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth

            return depth
        else:
            return invert_depth


def make_fusion_block(features, batch_norm):
    return FeatureFusionBlockCustom(
        features, nn.ReLU(), deconvolution=False, batch_norm=batch_norm, expand=False, align_corners=True)
