# encoding utf-8
import os.path

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import torch
import torch.nn.functional as func

from neural_renderer import ARNet, IUNet
from classical_renderer.scatter import ModuleRenderScatter


def gaussian_blur(input_data, radius, sigma=None):
    radius = int(round(radius))

    if sigma is None:
        sigma = 0.3 * (radius - 1) + 0.8
    # if end

    x_grid, y_grid = torch.meshgrid(torch.arange(-int(radius), int(radius) + 1),
                                    torch.arange(-int(radius), int(radius) + 1), indexing='ij')
    kernel0 = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
    kernel1 = kernel0.float() / kernel0.sum()
    kernel2 = kernel1.expand(1, 1, 2 * radius + 1, 2 * radius + 1).to(input_data.device)

    input_data = func.pad(input_data, pad=(radius, radius, radius, radius), mode='replicate')
    input_data = func.conv2d(input_data, weight=kernel2, padding=0)

    return input_data


def pipeline(classical_renderer, arnet, iunet, image, refocus, gamma, args):
    bokeh_classical, refocus_dilate = classical_renderer(image ** gamma, refocus * args.refocus_scale)

    adjusted_bokeh_classical = bokeh_classical ** (1 / gamma)
    adjusted_refocus_dilate = refocus_dilate / args.refocus_scale
    adjusted_gamma = (gamma - args.gamma_min) / (args.gamma_max - args.gamma_min)
    adjusted_scale = max(refocus.abs().max().item(), 1)

    rescale_image = func.interpolate(image, scale_factor=1/adjusted_scale, mode='bilinear', align_corners=True)
    rescale_refocus = (1 / adjusted_scale *
                       func.interpolate(refocus, scale_factor=1/adjusted_scale, mode='bilinear', align_corners=True))

    bokeh_neural, error_map = arnet(rescale_image, rescale_refocus, adjusted_gamma)
    rescale_error_map = func.interpolate(error_map, size=(image.shape[2], image.shape[3]),
                                         mode='bilinear', align_corners=True)

    bokeh_neural.clamp(0, 1e5)

    if args.save_intermediate:
        cv2.imwrite(os.path.join(str(save_root), 'bokeh_neural_s0.jpg'),
                    bokeh_neural[0].cpu().permute(1, 2, 0).numpy()[..., ::-1] * 255)
    # if end

    scale = -1

    for scale in range(int(np.log2(adjusted_scale))):
        ratio = 2 ** (scale + 1) / adjusted_scale
        resize_height, resize_weight = int(ratio * image.shape[2]), int(ratio * image.shape[3])
        rescale_image = func.interpolate(image, size=(resize_height, resize_weight),
                                         mode='bilinear', align_corners=True)
        rescale_refocus = ratio * func.interpolate(refocus, size=(resize_height, resize_weight),
                                                   mode='bilinear', align_corners=True)
        rescale_refocus_dilate = ratio * func.interpolate(adjusted_refocus_dilate, size=(resize_height, resize_weight),
                                                          mode='bilinear', align_corners=True)
        refined_bokeh_neural = (iunet(rescale_image, rescale_refocus.clamp(-1, 1), bokeh_neural, adjusted_gamma)
                                .clamp(0, 1e5))
        rescale_mask = gaussian_blur(((rescale_refocus_dilate < 1) * (rescale_refocus_dilate > -1)).float(),
                                     0.005 * (rescale_refocus_dilate.shape[2] + rescale_refocus_dilate.shape[3]))
        bokeh_neural = (rescale_mask * refined_bokeh_neural + (1 - rescale_mask) *
                        func.interpolate(bokeh_neural, size=(resize_height, resize_weight),
                                         mode='bilinear', align_corners=True))

        if args.save_intermediate:
            cv2.imwrite(os.path.join(str(save_root), f'bokeh_neural_s{scale+1}.jpg'),
                        bokeh_neural[0].cpu().permute(1, 2, 0).numpy()[..., ::-1] * 255)
            cv2.imwrite(os.path.join(str(save_root), f'fuse_mask_neural_s{scale+1}.jpg'),
                        rescale_mask[0][0].cpu().numpy() * 255)
        # if end

    refined_bokeh_neural = iunet(image, refocus.clamp(-1, 1), bokeh_neural, adjusted_gamma).clamp(0, 1e5)
    adjusted_mask = gaussian_blur(((adjusted_refocus_dilate < 1) * (adjusted_refocus_dilate > -1)).float(),
                                  0.005 * (refocus_dilate.shape[2] + refocus_dilate.shape[3]))
    bokeh_neural = (adjusted_mask * refined_bokeh_neural + (1 - adjusted_mask) *
                    func.interpolate(bokeh_neural, size=(image.shape[2], image.shape[3]),
                                     mode='bilinear', align_corners=True))

    if args.save_intermediate:
        cv2.imwrite(os.path.join(str(save_root), f'bokeh_neural_s{scale + 2}.jpg'),
                    bokeh_neural[0].cpu().permute(1, 2, 0).numpy()[..., ::-1] * 255)
        cv2.imwrite(os.path.join(str(save_root), f'fuse_mask_neural_s{scale + 2}.jpg'),
                    adjusted_mask[0][0].cpu().numpy() * 255)
    # if end

    bokeh_pred = adjusted_bokeh_classical * (1 - rescale_error_map) + bokeh_neural * rescale_error_map

    return bokeh_pred.clamp(0, 1), adjusted_bokeh_classical.clamp(0, 1), bokeh_neural.clamp(0, 1), rescale_error_map


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Bokeh Rendering', fromfile_prefix_chars='@')

parser.add_argument('--refocus_scale',               type=float, default=10.0)
parser.add_argument('--gamma_min',                   type=float, default=1.0)
parser.add_argument('--gamma_max',                   type=float, default=5.0)

# ARNet argument
parser.add_argument('--arnet_shuffle_rate',          type=int, default=2)
parser.add_argument('--arnet_in_channels',           type=int, default=5)
parser.add_argument('--arnet_out_channels',          type=int, default=4)
parser.add_argument('--arnet_middle_channels',       type=int,   default=128)
parser.add_argument('--arnet_num_block',             type=int,   default=3)
parser.add_argument('--arnet_share_weight',                      action='store_true')
parser.add_argument('--arnet_connect_mode',          type=str,   default='distinct_source')
parser.add_argument('--arnet_batch_norm',                            action='store_true')
parser.add_argument('--arnet_activation',            type=str,   default='elu')

# IUNet argument
parser.add_argument('--iunet_shuffle_rate',          type=int,   default=2)
parser.add_argument('--iunet_in_channels',           type=int,   default=8)
parser.add_argument('--iunet_out_channels',          type=int,   default=3)
parser.add_argument('--iunet_middle_channels',       type=int,   default=64)
parser.add_argument('--iunet_num_block',             type=int,   default=3)
parser.add_argument('--iunet_share_weight',                      action='store_true')
parser.add_argument('--iunet_connect_mode',          type=str,   default='distinct_source')
parser.add_argument('--iunet_batch_norm',                            action='store_true')
parser.add_argument('--iunet_activation',            type=str,   default='elu')

# Checkpoint
parser.add_argument('--arnet_checkpoint_path',       type=str,   default='./checkpoints/arnet.pth')
parser.add_argument('--iunet_checkpoint_path',       type=str,   default='./checkpoints/iunet.pth')

# Input
parser.add_argument('--image_path',                  type=str,   default='./inputs/21.jpg')
parser.add_argument('--disp_path',                   type=str,   default='./inputs/21.png')
parser.add_argument('--save_dir',                    type=str,   default='./outputs')
parser.add_argument('--K',                           type=float, default=60,          help='blur parameter')
parser.add_argument('--disp_focus',                  type=float, default=90/255,      help='refocused disparity (0~1)')
parser.add_argument('--gamma',                       type=float, default=4,           help='gamma value (1~5)')

parser.add_argument('--highlight',   action='store_true', help='forcibly enhance RGB values of highlights')
parser.add_argument('--highlight_RGB_threshold',     type=float, default=220/255)
parser.add_argument('--highlight_enhance_ratio',     type=float, default=0.4)

parser.add_argument('--save_intermediate',                       action='store_true', help='save intermediate results')

parser_args = parser.parse_args()

arnet_checkpoint_path = parser_args.arnet_checkpoint_path
iunet_checkpoint_path = parser_args.iunet_checkpoint_path

classical_model = ModuleRenderScatter().to(device)

arnet_model = ARNet(parser_args.arnet_shuffle_rate, parser_args.arnet_in_channels, parser_args.arnet_out_channels,
                    parser_args.arnet_middle_channels, parser_args.arnet_num_block, parser_args.arnet_share_weight,
                    parser_args.arnet_connect_mode, parser_args.arnet_batch_norm, parser_args.arnet_activation)
iunet_model = IUNet(parser_args.iunet_shuffle_rate, parser_args.iunet_in_channels, parser_args.iunet_out_channels,
                    parser_args.iunet_middle_channels, parser_args.iunet_num_block, parser_args.iunet_share_weight,
                    parser_args.iunet_connect_mode, parser_args.iunet_batch_norm, parser_args.iunet_activation)

arnet_model.cuda()
iunet_model.cuda()

arnet_checkpoint = torch.load(arnet_checkpoint_path)
arnet_model.load_state_dict(arnet_checkpoint['model'])
iunet_checkpoint = torch.load(iunet_checkpoint_path)
iunet_model.load_state_dict(iunet_checkpoint['model'])

arnet_model.eval()
iunet_model.eval()

save_root = os.path.join(parser_args.save_dir, os.path.splitext(os.path.basename(parser_args.image_path))[0])

os.makedirs(save_root, exist_ok=True)

k_value = parser_args.K                   # blur parameter
disp_focus_value = parser_args.disp_focus  # 0 ~ 1
gamma_value = parser_args.gamma           # 1 ~ 5

image_data = cv2.imread(parser_args.image_path).astype(np.float32) / 255.0
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
origin_image = image_data.copy()

disp = np.float32(cv2.imread(parser_args.disp_path, cv2.IMREAD_GRAYSCALE))
disp = (disp - disp.min()) / (disp.max() - disp.min())


# --------------------- Highlight --------------------- #

if parser_args.highlight:
    mask1 = np.clip(np.tanh(200 * (np.abs(disp - disp_focus_value) ** 2 - 0.01)), 0, 1)[..., np.newaxis]
    mask2 = np.clip(np.tanh(10 * (image_data - parser_args.highlight_RGB_threshold)), 0, 1)

    mask = mask1 * mask2
    image_data = image_data * (1 + mask * parser_args.highlight_enhance_ratio)

# ----------------------------------------------------- #

refocus_value = k_value * (disp - disp_focus_value) / parser_args.refocus_scale

with torch.no_grad():
    image_data = torch.from_numpy(image_data).permute(2, 0, 1).unsqueeze(0)
    refocus_value = torch.from_numpy(refocus_value).unsqueeze(0).unsqueeze(0)
    image_data = image_data.cuda()
    refocus_value = refocus_value.cuda()

    bokeh_pred_result, bokeh_classical_result, bokeh_neural_result, error_map_result = pipeline(
        classical_model, arnet_model, iunet_model, image_data, refocus_value, gamma_value, parser_args)

refocus_value = refocus_value[0][0].cpu().numpy()
error_map_result = error_map_result[0][0].cpu().numpy()
bokeh_classical_result = bokeh_classical_result[0].cpu().permute(1, 2, 0).numpy()
bokeh_neural_result = bokeh_neural_result[0].cpu().permute(1, 2, 0).detach().numpy()
bokeh_pred_result = bokeh_pred_result[0].cpu().permute(1, 2, 0).detach().numpy()

cv2.imwrite(os.path.join(str(save_root), 'image.jpg'), origin_image[..., ::-1] * 255)
plt.imsave(os.path.join(str(save_root), 'refocus.jpg'), refocus_value, cmap='coolwarm',
           vmin=-max(refocus_value.max(), -refocus_value.min()), vmax=max(refocus_value.max(), -refocus_value.min()))
cv2.imwrite(os.path.join(str(save_root), 'disparity.jpg'), disp * 255)
cv2.imwrite(os.path.join(str(save_root), 'error_map.jpg'), error_map_result * 255)
cv2.imwrite(os.path.join(str(save_root), 'bokeh_classical.jpg'), bokeh_classical_result[..., ::-1] * 255)
cv2.imwrite(os.path.join(str(save_root), 'bokeh_neural.jpg'), bokeh_neural_result[..., ::-1] * 255)
cv2.imwrite(os.path.join(str(save_root), 'bokeh_pred_result.jpg'), bokeh_pred_result[..., ::-1] * 255)
