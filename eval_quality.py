import cv2
import torch
import torch.nn.functional as func


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize to [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def calculate_psnr(img1, img2):
    mse = func.mse_loss(img1, img2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


classic_bokeh = load_image('outputs/47/bokeh_classical.jpg')
neural_bokeh = load_image('outputs/47/bokeh_neural.jpg')
pred_bokeh = load_image('outputs/47/bokeh_pred_result.jpg')
origin_image = load_image('outputs/47/image.jpg')

psnr_classic = calculate_psnr(classic_bokeh, origin_image)
psnr_neural = calculate_psnr(neural_bokeh, origin_image)
psnr_pred = calculate_psnr(pred_bokeh, origin_image)

print(f"PSNR for Classic Bokeh: {psnr_classic:.2f} dB")
print(f"PSNR for Neural Bokeh: {psnr_neural:.2f} dB")
print(f"PSNR for Pred Bokeh: {psnr_pred:.2f} dB")
