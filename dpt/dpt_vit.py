import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as func


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, data):
        return data[:, self.start_index:]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, data):
        if self.start_index == 2:
            readout = (data[:, 0] + data[:, 1]) / 2
        else:
            readout = data[:, 0]

        return data[:, self.start_index:] + readout.unsqueeze(1)


class ProjectionReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectionReadout, self).__init__()
        self.start_index = start_index
        self.projection = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, data):
        readout = data[:, 0].unsqueeze(1).expand_as(data[:, self.start_index:])
        features = torch.cat((data[:, self.start_index:], readout), -1)

        return self.projection(features)


class Transpose(nn.Module):
    def __init__(self, first_dim, second_dim):
        super(Transpose, self).__init__()
        self.first_dim = first_dim
        self.second_dim = second_dim

    def forward(self, data):
        return data.transpose(self.first_dim, self.second_dim)


def get_activation(name):
    def hook(_, __, outputs):  # model, inputs, outputs
        activations[name] = outputs

    return hook


def get_attention(name):
    def hook(module, inputs, _):  # module, inputs, outputs
        data = inputs[0]
        batch, num_patch, channel = data.shape
        qkv = (module.qkv(data)
               .reshape(batch, num_patch, 3, module.num_heads, channel // module.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = (qkv[0], qkv[1], qkv[2])

        attention_value = (q @ k.transpose(-2, -1)) * module.scale
        attention_value = attention_value.softmax(dim=-1)
        attention[name] = attention_value

    return hook


def resize_pos_embedding(self, pos_embedding, grid_size_height, grid_size_width):
    pos_embedding_token, pos_embedding_grid = (pos_embedding[:, :self.start_index], pos_embedding[0, self.start_index:])
    grid_size_old = int(math.sqrt(len(pos_embedding_grid)))

    pos_embedding_grid = pos_embedding_grid.reshape(1, grid_size_old, grid_size_old, -1).permute(0, 3, 1, 2)
    pos_embedding_grid = func.interpolate(pos_embedding_grid, size=(grid_size_height, grid_size_width), mode='bilinear')
    pos_embedding_grid = pos_embedding_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)

    pos_embedding = torch.cat([pos_embedding_token, pos_embedding_grid], dim=1)

    return pos_embedding


def forward_flex(self, data):
    batch, channel, height, width = data.shape
    pos_embed = self.resize_pos_embedding(self.pos_embedding, height // self.patch_size[1], width // self.patch_size[0])
    # B = data.shape[0]

    if hasattr(self.patch_embed, 'backbone'):
        data = self.patch_embed.backbone(data)

        if isinstance(data, (list, tuple)):
            data = data[-1]

    data = self.patch_embed.proj(data).flatten(2).transpose(1, 2)

    if getattr(self, 'dist_token', None) is not None:
        class_token = self.class_token.expand(batch, -1, -1)
        dist_token = self.dist_token.expand(batch, -1, -1)
        data = torch.cat((class_token, dist_token, data), dim=1)
    else:
        class_tokens = self.class_token.expand(batch, -1, -1)
        data = torch.cat((class_tokens, data), dim=1)

    data = data + pos_embed
    data = self.pos_dropout(data)

    for block in self.blocks:
        data = block(data)

    data = self.norm(data)

    return data


def forward_vit(pretrained, data):
    batch, channel, height, width = data.shape
    _ = pretrained.model.forward_flex(data)

    layer1 = pretrained.activations['1']
    layer2 = pretrained.activations['2']
    layer3 = pretrained.activations['3']
    layer4 = pretrained.activations['4']

    layer1 = pretrained.act_postprocess1[0:2](layer1)
    layer2 = pretrained.act_postprocess2[0:2](layer2)
    layer3 = pretrained.act_postprocess3[0:2](layer3)
    layer4 = pretrained.act_postprocess4[0:2](layer4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2, torch.Size([height // pretrained.model.patch_size[1], width // pretrained.model.patch_size[0]])
        )
    )

    if layer1.ndim == 3:
        layer1 = unflatten(layer1)
    if layer1.ndim == 3:
        layer2 = unflatten(layer2)
    if layer1.ndim == 3:
        layer3 = unflatten(layer3)
    if layer1.ndim == 3:
        layer4 = unflatten(layer4)

    layer1 = pretrained.act_postprocess1[3: len(pretrained.act_postprocess1)](layer1)
    layer2 = pretrained.act_postprocess2[3: len(pretrained.act_postprocess2)](layer2)
    layer3 = pretrained.act_postprocess1[3: len(pretrained.act_postprocess3)](layer3)
    layer4 = pretrained.act_postprocess1[3: len(pretrained.act_postprocess4)](layer4)

    return layer1, layer2, layer3, layer4


def get_readout_operation(vit_features, features, use_readout, start_index=1):
    if use_readout == 'ignore':
        readout_operation = [Slice(start_index)] * len(features)
    elif use_readout == 'add':
        readout_operation = [AddReadout(start_index)] * len(features)
    elif use_readout == 'projection':
        readout_operation = [ProjectionReadout(vit_features, start_index) for _ in features]  # out_feature
    else:
        assert False, "wrong operation for readout_token, use_readout can be 'ignore', 'add', 'projection'"

    return readout_operation


def make_vit_base_resnet50_backbone(
    model,
    features=None,
    size=None,
    hooks=None,
    vit_features=768,
    use_vit_only=False,
    use_readout='ignore',
    start_index=1,
    enable_attention_hooks=False
):
    features = [96, 192, 384, 768] if features is None else features
    size = [384, 384] if size is None else size
    hooks = [2, 5, 8, 11] if hooks is None else hooks

    pretrained = nn.Module()
    pretrained.model = model

    if use_vit_only:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation('1'))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation('2'))
    else:
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(get_activation('1'))
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(get_activation('2'))

    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation('3'))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation('4'))

    if enable_attention_hooks:
        pretrained.model.blocks[2].attention.register_forward_hook(get_attention('attention_1'))
        pretrained.model.blocks[5].attention.register_forward_hook(get_attention('attention_2'))
        pretrained.model.blocks[8].attention.register_forward_hook(get_attention('attention_3'))
        pretrained.model.blocks[11].attention.register_forward_hook(get_attention('attention_3'))
        pretrained.attention = attention

    pretrained.activations = activations

    readout_operation = get_readout_operation(vit_features, features, use_readout, start_index)

    if use_vit_only:
        pretrained.act_post_process1 = nn.Sequential(
            readout_operation[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1
            )
        )

        pretrained.act_post_process2 = nn.Sequential(
            readout_operation[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1
            )
        )
    else:
        pretrained.act_postprocess1 = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity())
        pretrained.act_postprocess2 = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity())

    pretrained.act_post_process3 = nn.Sequential(
        readout_operation[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0
        )
    )
    pretrained.act_post_process4 = nn.Sequential(
        readout_operation[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1
        )
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model.resize_pos_embedding = types.MethodType(resize_pos_embedding, pretrained.model)

    return pretrained


def make_pretrained_vit_base_resnet50_384(
    pretrained, use_readout='ignore', hooks=None, use_vit_only=False, enable_attention_hooks=False
):
    model = timm.create_model('vit_base_resnet50_384', pretrained=pretrained)

    hooks = [0, 1, 8, 11] if hooks is None else hooks

    return make_vit_base_resnet50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks
    )


activations = {}
attention = {}
