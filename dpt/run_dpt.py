import torch
import torch.nn.functional as func

from torchvision.transforms import Compose
from .dpt_io import read_image, write_depth
from .dpt_model import DPTDepthModel
from .dpt_transforms import *  # Resize, NormalizeImage, PrepareForNet


def set_model_type(model) -> torch.Tensor:
    return model


def dpt_run(input_path, model_path, optimize=True):
    print('dpt initialize')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('current device: %s' % device)

    model_width, model_height = 384, 384
    model = DPTDepthModel(path=model_path, backbone='vit_base_resnet50_384',
                          non_negative=True, enable_attention_hooks=False)
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose([
        Resize(model_width, model_height, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
               resize_method='minimal', interpolation_method=cv2.INTER_CUBIC),
        normalization,
        PrepareForNet()
    ])

    model.eval()

    if optimize and device == torch.device('cuda'):
        # set model type DPTDepthModel to Tensor to express
        model = set_model_type(model)
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    print('start dpt processing')

    img = read_image(input_path)

    img_input = transform({'image': img})['image']

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        if optimize and device == torch.device('cuda'):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            func.interpolate(prediction.unsqueeze(1), size=img.shape[:2], mode='bicubic', align_corners=False)
            .squeeze().cpu().numpy()
        )

        file_name = input_path[:-4]
        write_depth(file_name, prediction, bits=2, absolute_depth=False)
    print('dpt is finished')
