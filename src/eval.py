import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

def load_img(path):
    img = Image.open(path).convert('L').resize((80,80))
    return np.array(img).astype(np.float32)

def eval_pair(ref_path, pred_path):
    a = load_img(ref_path)
    b = load_img(pred_path)
    s = ssim(a, b, data_range=255)
    p = psnr(a, b, data_range=255)
    return s, p

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True)
    parser.add_argument('--pred', required=True)
    args = parser.parse_args()
    s,p = eval_pair(args.ref, args.pred)
    print(f'SSIM={s:.4f}, PSNR={p:.4f}')
