import os
import torch
from neural_renderer import ARNet, IUNet
from torchsummary import summary
from dpt.dpt_model import DPTDepthModel

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_summary(is_dpt=False):
    arnet_model = ARNet().to(device)
    iunet_model = IUNet().to(device)

    print("ARNet Summary")
    summary(arnet_model, [(3, 1920, 1080), (1, 1920, 1080), (1, 1, 1)], batch_size=16)

    print("IUNet Summary")
    summary(iunet_model, [(3, 1920, 1080), (1, 1920, 1080), (3, 1920, 1080), (1, 1, 1)], batch_size=16)

    if is_dpt:
        dpt_model = DPTDepthModel(path='weights/dpt_hybrid-midas-501f0c75.pt', backbone='vit_base_resnet50_384',
                                  non_negative=True, enable_attention_hooks=False).to(device)

        print("DPT Summary")
        summary(dpt_model, (3, 384, 384), batch_size=16)
