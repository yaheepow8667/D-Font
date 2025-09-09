# DP-Font Repro - src/ (All core files)
# Each section below is a separate file. Save them under the paths shown in the header comments.

# ---------- File: src/dataset.py ----------
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class DPFontDataset(Dataset):
    """Dataset expects images organized as: image_root/FontName/00020.png
    and stroke_root/00020.npy containing (36,) int array.
    char_list is a list of ints (unicode codepoints) to include.
    """
    def __init__(self, image_root, stroke_root, font_list, char_list, transform=None):
        self.image_root = image_root
        self.stroke_root = stroke_root
        self.font_list = font_list
        self.char_list = char_list
        self.transform = transform

        self.samples = []
        for font in font_list:
            for char_code in char_list:
                img_path = os.path.join(image_root, font, f"{char_code:05d}.png")
                stroke_path = os.path.join(stroke_root, f"{char_code:05d}.npy")
                if os.path.exists(img_path) and os.path.exists(stroke_path):
                    self.samples.append((img_path, stroke_path, font, char_code))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, stroke_path, font_name, char_code = self.samples[idx]
        img = Image.open(img_path).convert('L').resize((80,80))
        arr = np.array(img).astype(np.float32)
        # normalize to [-1,1]
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0
        stroke = torch.from_numpy(np.load(stroke_path)).long()  # (36,)
        style_id = torch.tensor(abs(hash(font_name)) % 200, dtype=torch.long)
        content_id = torch.tensor(char_code, dtype=torch.long)
        return {'image': tensor, 'stroke': stroke, 'style': style_id, 'content': content_id}


# ---------- File: src/attribute_encoder.py ----------
import torch
import torch.nn as nn

class AttributeEncoder(nn.Module):
    def __init__(self, content_dim=128, stroke_dim=128, style_dim=128,
                 output_dim=512, max_strokes=36, num_chars=20902, num_styles=200):
        super().__init__()
        self.content_embed = nn.Embedding(num_chars, content_dim)
        self.stroke_embed = nn.Embedding(6, 16)  # 0-5 mapping
        self.stroke_mlp = nn.Sequential(
            nn.Linear(max_strokes * 16, 256),
            nn.ReLU(),
            nn.Linear(256, stroke_dim)
        )
        self.style_embed = nn.Embedding(num_styles, style_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(content_dim + stroke_dim + style_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, content_ids, stroke_orders, style_ids):
        c = self.content_embed(content_ids)
        s = self.stroke_embed(stroke_orders)  # [B,36,16]
        s = s.view(s.size(0), -1)
        s = self.stroke_mlp(s)
        sty = self.style_embed(style_ids)
        z = self.fusion_mlp(torch.cat([c, s, sty], dim=1))
        return z


# ---------- File: src/pinn.py ----------
import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNLoss(nn.Module):
    def __init__(self, D=0.1, loss_type='mse', sign_mode='plus'):
        """sign_mode: 'plus' means resid = du_dt + D * laplacian
           'minus' means resid = du_dt - D * laplacian
           'laplace_only' uses only laplacian(u)
        """
        super().__init__()
        self.D = D
        self.loss_type = loss_type
        self.sign_mode = sign_mode
        kernel = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kernel', kernel)

    def compute_lap(self, u):
        B,C,H,W = u.shape
        k = self.kernel.repeat(C,1,1,1)
        u_pad = F.pad(u, (1,1,1,1), mode='reflect')
        lap = F.conv2d(u_pad, k, groups=C)
        return lap

    def forward(self, u_t, u_t_minus_1=None):
        # u_t: [B,C,H,W]
        if self.sign_mode == 'laplace_only' or u_t_minus_1 is None:
            lap = self.compute_lap(u_t)
            resid = lap
        else:
            du_dt = (u_t - u_t_minus_1)
            lap = self.compute_lap(u_t)
            if self.sign_mode == 'minus':
                resid = du_dt - self.D * lap
            else:
                resid = du_dt + self.D * lap
        if self.loss_type == 'mse':
            return F.mse_loss(resid, torch.zeros_like(resid))
        else:
            return F.l1_loss(resid, torch.zeros_like(resid))


# ---------- File: src/model/dp_unet.py ----------
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyResBlock(nn.Module):
    def __init__(self, channels, emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        if emb_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, channels*2))
        else:
            self.mlp = None

    def forward(self, x, emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        if self.mlp is not None and emb is not None:
            scale_shift = self.mlp(emb)
            scale, shift = scale_shift.chunk(2, dim=1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale) + shift
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h

class SimpleUNet(nn.Module):
    """A small UNet for 80x80 grayscale images. Accepts conditioning vector z (B,512).
       Time embedding t_emb should be passed in as well.
    """
    def __init__(self, base_ch=64, z_dim=512, t_dim=128):
        super().__init__()
        self.init_conv = nn.Conv2d(1, base_ch, 3, padding=1)
        self.down1 = nn.Sequential(TinyResBlock(base_ch, emb_dim=z_dim+t_dim), nn.AvgPool2d(2))
        self.down2 = nn.Sequential(TinyResBlock(base_ch*2, emb_dim=z_dim+t_dim), nn.AvgPool2d(2))
        self.mid = TinyResBlock(base_ch*4, emb_dim=z_dim+t_dim)
        self.up2 = nn.Sequential(TinyResBlock(base_ch*2, emb_dim=z_dim+t_dim))
        self.up1 = nn.Sequential(TinyResBlock(base_ch, emb_dim=z_dim+t_dim))
        self.final = nn.Conv2d(base_ch, 1, 3, padding=1)

        # simple channel adaptors
        self.increase1 = nn.Conv2d(base_ch, base_ch*2, 1)
        self.increase2 = nn.Conv2d(base_ch*2, base_ch*4, 1)
        self.decrease2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.decrease1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)

    def forward(self, x, t_emb, z):
        # x: [B,1,80,80]; t_emb: [B,t_dim]; z: [B,z_dim]
        emb = torch.cat([z, t_emb], dim=1)
        h0 = self.init_conv(x)  # [B,base,80,80]
        h1 = self.down1[0](h0, emb)
        h1p = self.down1[1](h1)
        h2 = self.increase1(h1p)
        h2 = self.down2[0](h2, emb)
        h2p = self.down2[1](h2)
        h3 = self.increase2(h2p)
        h3 = self.mid(h3, emb)
        u2 = self.decrease2(h3)
        u2 = self.up2[0](u2 + h2, emb)
        u1 = self.decrease1(u2)
        u1 = self.up1[0](u1 + h1, emb)
        out = self.final(u1)
        return out

# small helper for time embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        # t: [B] int tensor
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.size(0),1, device=t.device)], dim=1)
        return self.proj(emb)


# ---------- File: src/model/dp_model.py ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dp_unet import SimpleUNet, TimeEmbedding
from attribute_encoder import AttributeEncoder

class DPFontModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.attr = AttributeEncoder()
        self.t_emb = TimeEmbedding(128)
        self.unet = SimpleUNet(base_ch=32, z_dim=512, t_dim=128)
        self.device = device

    def forward(self, x_t, t, content_ids, stroke_orders, style_ids):
        # x_t: [B,1,H,W], t: [B] ints
        z = self.attr(content_ids, stroke_orders, style_ids)
        t_emb = self.t_emb(t)
        pred_eps = self.unet(x_t, t_emb, z)
        return pred_eps

    @torch.no_grad()
    def sample(self, content_ids, stroke_orders, style_ids, T=1000, guidance_scale=3.0, device='cuda'):
        B = content_ids.size(0)
        x = torch.randn(B,1,80,80, device=device)
        # simple linear betas
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        for i in reversed(range(T)):
            t = torch.tensor([i]*B, device=device)
            # conditional
            z_cond = self.attr(content_ids, stroke_orders, style_ids)
            t_emb = self.t_emb(t)
            eps_cond = self.unet(x, t_emb, z_cond)
            # uncond: zeroed stroke/style
            z_uncond = self.attr(content_ids, torch.zeros_like(stroke_orders), torch.zeros_like(style_ids))
            eps_uncond = self.unet(x, t_emb, z_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            alpha_t = alphas[i]
            alpha_bar = alpha_bars[i]
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar)
            mu = coef1 * (x - coef2 * eps)
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(betas[i])
                x = mu + sigma * noise
            else:
                x = mu
        return x


# ---------- File: src/utils.py ----------
import torch
import os
from torchvision.utils import save_image

def save_batch(img_tensor, out_dir, prefix='sample', nrow=8):
    os.makedirs(out_dir, exist_ok=True)
    # img_tensor expected in [-1,1]
    t = (img_tensor.clamp(-1,1) + 1.0) / 2.0
    save_image(t, os.path.join(out_dir, f"{prefix}.png"), nrow=nrow)

# simple beta schedule
def make_alphas(T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


# ---------- File: src/train.py ----------
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from attribute_encoder import AttributeEncoder
from dataset import DPFontDataset
from pinn import PINNLoss
from model.dp_model import DPFontModel
from utils import make_alphas, save_batch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--stroke_root', type=str, default='data/stroke_order')
parser.add_argument('--fonts', type=str, default='FontA,FontB')
parser.add_argument('--chars', type=str, default='20013')
parser.add_argument('--out', type=str, default='out')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--lambda_pinn', type=float, default=0.1)
parser.add_argument('--p_uncond', type=float, default=0.1)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
font_list = args.fonts.split(',')
char_list = [int(x) for x in args.chars.split(',')]

dataset = DPFontDataset(os.path.join(args.data_root,'images'), args.stroke_root, font_list, char_list)
dloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

model = DPFontModel(device=device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

T = 1000
betas, alphas, alpha_bars = make_alphas(T=T, device=device)

pinn = PINNLoss(D=0.1, sign_mode='plus').to(device)

global_step = 0
for epoch in range(args.epochs):
    for batch in dloader:
        x0 = batch['image'].to(device)  # [-1,1]
        content = batch['content'].to(device)
        stroke = batch['stroke'].to(device)
        style = batch['style'].to(device)

        # sample t and noise
        t = torch.randint(1, T, (x0.size(0),), device=device)
        eps = torch.randn_like(x0)
        sqrt_ab = torch.sqrt(alpha_bars[t])[:,None,None,None]
        sqrt_1_ab = torch.sqrt(1 - alpha_bars[t])[:,None,None,None]
        x_t = sqrt_ab * x0 + sqrt_1_ab * eps

        # classifier free mask
        if torch.rand(1).item() < args.p_uncond:
            stroke_in = torch.zeros_like(stroke)
            style_in = torch.zeros_like(style)
        else:
            stroke_in = stroke
            style_in = style

        pred_eps = model.forward(x_t, t, content, stroke_in, style_in)
        loss_simple = F.mse_loss(pred_eps, eps)

        # approx x_{t-1}
        alpha_t = alphas[t][:,None,None,None]
        alpha_bar = alpha_bars[t][:,None,None,None]
        mu = (1/torch.sqrt(alpha_t)) * (x_t - ((1-alpha_t)/torch.sqrt(1-alpha_bar)) * pred_eps)
        x_t_minus_1_approx = mu.detach()

        # compute x0_pred
        sqrt_at = torch.sqrt(alpha_t)
        sqrt_1_at = torch.sqrt(1 - alpha_bar)
        x0_pred = (x_t - sqrt_1_at * pred_eps) / (sqrt_at + 1e-8)

        # PINN: using laplacian on x0_pred as default stable choice
        loss_pinn = pinn(x0_pred)

        loss = loss_simple + args.lambda_pinn * loss_pinn

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if global_step % 100 == 0:
            print(f"step {global_step}: L_simple={loss_simple.item():.6f} L_pinn={loss_pinn.item():.6f} total={loss.item():.6f}")

        if global_step % 1000 == 0:
            with torch.no_grad():
                sample_x = model.sample(content[:8].to(device), stroke[:8].to(device), style[:8].to(device), T=100, guidance_scale=3.0, device=device)
                save_batch(sample_x, args.out, prefix=f'step_{global_step}')

        global_step += 1

    # save checkpoint per epoch
    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict()}, os.path.join(args.out, f'ckpt_epoch_{epoch}.pt'))

print('Training finished')


# ---------- File: src/sample.py ----------
import torch
from model.dp_model import DPFontModel
from utils import save_batch

def run_sample(ckpt_path, content_list, stroke_list, style_list, out_dir='samples'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DPFontModel(device=device).to(device)
    data = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(data['model'])
    content = torch.tensor(content_list, dtype=torch.long, device=device)
    stroke = torch.tensor(stroke_list, dtype=torch.long, device=device)
    style = torch.tensor(style_list, dtype=torch.long, device=device)
    with torch.no_grad():
        x = model.sample(content, stroke, style, T=100, guidance_scale=3.5, device=device)
        save_batch(x, out_dir, prefix='sample')


# ---------- File: src/eval.py ----------
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

# End of src/ files


# ---------- File: requirements.txt (expanded) ----------
# Use this file to create a virtual environment and install minimal dependencies
# Adjust torch/cu version per your CUDA setup (example requires a compatible torch build).
torch
torchvision
numpy
Pillow
scikit-image
tqdm

# ---------- File: run.sh ----------
# Example run script: train (toy), fine-tune (few-shot) and sample. Make executable: chmod +x run.sh
#!/usr/bin/env bash
set -e

echo "=== Create and activate virtualenv ==="
python -m venv venv_dpfont
source venv_dpfont/bin/activate

echo "=== Upgrade pip and install requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Quick training run (toy) ==="
# Make sure you have data/images and data/stroke_order prepared as described in README.md
python src/train.py --data_root data --stroke_root data/stroke_order --fonts FontA,FontB --chars 20013 --epochs 1 --batch 8 --out out

echo "=== Fine-tune (few-shot) example ==="
# Example: fine-tune using a style reference set (the calligrapher's 200 images).
# - style_root should contain images named by codepoint (e.g. 20013.png) or use src/prepare_style_refs.py to map.
# - base_root: optional root of base images (many-font rendered dataset) to help content learning.
# - pretrained_ckpt: optional path to a pretrained checkpoint to load before fine-tuning.
# Run with --augment to enable augmentations and --use_pinn to enable PINN loss (if desired).
python src/fine_tune.py \
  --base_root data/images \
  --style_root data/style_refs \
  --stroke_root data/stroke_order \
  --char_list 20013,22269 \
  --pretrained_ckpt '' \
  --out out_finetune \
  --epochs 3 \
  --batch 4 \
  --K 5 \
  --augment \
  --T 1000 \
  --lr 2e-5

echo "=== Sampling example (after checkpoint is saved) ==="
# Adjust checkpoint path as necessary. The sample helper is a function; here we call it via Python -c.
CKPT=out/ckpt_epoch_0.pt
python - <<PY
from src.sample import run_sample
# single example: content codepoint 20013, stroke list as zeros (36,), style id 0
run_sample(CKPT, [20013], [[0]*36], [0], out_dir='samples')
PY

echo "Done. Samples saved in ./samples"

# ---------- File: src/augment.py ----------
```python
# Augmentation utilities for DP-Font few-shot fine-tuning
# Provides functions to perform on-the-fly augmentations for the 200 sample set.
# Uses: PIL, numpy, cv2
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2


def to_numpy_gray(img):
    # img: PIL Image (L) -> np uint8
    return np.array(img)


def from_numpy_gray(arr):
    return Image.fromarray(arr.astype(np.uint8))


def random_rotate(img, max_angle=4.0):
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)


def random_translate_scale(img, max_shift=3, scale_range=(0.95,1.05)):
    w,h = img.size
    tx = random.uniform(-max_shift, max_shift)
    ty = random.uniform(-max_shift, max_shift)
    scale = random.uniform(scale_range[0], scale_range[1])
    M = np.array([[scale, 0, tx],[0, scale, ty]], dtype=np.float32)
    arr = to_numpy_gray(img)
    dst = cv2.warpAffine(arr, M, (w,h), borderMode=cv2.BORDER_REFLECT)
    return from_numpy_gray(dst)


def gaussian_blur(img, sigma=0.7):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def random_noise(img, var=5.0):
    arr = to_numpy_gray(img).astype(np.float32)
    noise = np.random.normal(0, var, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return from_numpy_gray(arr)


def dilate_erode(img, mode='dilate', ksize=1):
    arr = to_numpy_gray(img)
    kernel = np.ones((ksize,ksize), np.uint8)
    if mode == 'dilate':
        out = cv2.dilate(arr, kernel, iterations=1)
    else:
        out = cv2.erode(arr, kernel, iterations=1)
    return from_numpy_gray(out)


def elastic_transform(img, alpha=34, sigma=4):
    # simple elastic transform using OpenCV remap
    arr = to_numpy_gray(img)
    shape = arr.shape
    dx = (cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (17,17), sigma) * alpha).astype(np.float32)
    dy = (cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (17,17), sigma) * alpha).astype(np.float32)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return from_numpy_gray(warped)


def augment_image(img_pil, prob_config=None):
    """Apply a random sequence of augmentations to a PIL Image (L mode).
    prob_config: dict to control probabilities
    """
    if prob_config is None:
        prob_config = {}
    img = img_pil
    if random.random() < prob_config.get('rotate', 0.6):
        img = random_rotate(img, max_angle=prob_config.get('max_angle', 4.0))
    if random.random() < prob_config.get('translate', 0.6):
        img = random_translate_scale(img, max_shift=prob_config.get('max_shift', 3))
    if random.random() < prob_config.get('elastic', 0.3):
        img = elastic_transform(img, alpha=prob_config.get('alpha', 30), sigma=prob_config.get('sigma', 4))
    if random.random() < prob_config.get('blur', 0.2):
        img = gaussian_blur(img, sigma=prob_config.get('blur_sigma', 0.6))
    if random.random() < prob_config.get('morph', 0.2):
        img = dilate_erode(img, mode=random.choice(['dilate','erode']), ksize=random.choice([1,2]))
    if random.random() < prob_config.get('noise', 0.2):
        img = random_noise(img, var=prob_config.get('noise_var', 5.0))
    return img
```

# ---------- File: src/fine_tune.py ----------
```python
# fine_tune.py
# Few-shot style adaptation: style encoder (N-shot) + fine-tune UNet

import os
import random
import argparse
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model.dp_model import DPFontModel
from attribute_encoder import AttributeEncoder
from pinn import PINNLoss
from utils import make_alphas
from augment import augment_image


class StyleEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: [B,1,H,W]
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class FewShotDataset(Dataset):
    """Dataset for fine-tuning. If base_root provided, uses base images for content mapping.
    style_root contains many style reference images (200 images). We sample K per example.
    If base_root is None, we fall back to using style images as content images (not ideal).
    """
    def __init__(self, base_root, stroke_root, style_root, char_list, K=5, augment=False):
        self.base_root = base_root
        self.stroke_root = stroke_root
        self.style_root = style_root
        self.K = K
        self.augment = augment
        # gather base samples (pairs of (img_path, char_code))
        self.samples = []
        if base_root is not None and os.path.exists(base_root):
            for font in os.listdir(base_root):
                font_dir = os.path.join(base_root, font)
                for fn in os.listdir(font_dir):
                    if fn.endswith('.png'):
                        code = int(os.path.splitext(fn)[0])
                        self.samples.append((os.path.join(font_dir, fn), code))
        else:
            # fallback: use style images as content images
            for fn in os.listdir(style_root):
                if fn.endswith('.png'):
                    code = int(os.path.splitext(fn)[0]) if fn.split('.')[0].isdigit() else 0
                    self.samples.append((os.path.join(style_root, fn), code))
        # style pool
        self.style_pool = [os.path.join(style_root, fn) for fn in os.listdir(style_root) if fn.endswith('.png')]
        self.char_list = char_list

    def __len__(self):
        return max(1, len(self.samples))

    def __getitem__(self, idx):
        img_path, code = self.samples[idx % len(self.samples)]
        img = Image.open(img_path).convert('L').resize((80,80))
        if self.augment:
            img = augment_image(img)
        img_arr = (np.array(img).astype(np.float32)/127.5 - 1.0)
        img_t = torch.tensor(img_arr).unsqueeze(0)
        # content id
        content = torch.tensor(code, dtype=torch.long)
        # stroke vector
        stroke_path = os.path.join(self.stroke_root, f"{code:05d}.npy")
        if os.path.exists(stroke_path):
            stroke = torch.from_numpy(np.load(stroke_path)).long()
        else:
            stroke = torch.zeros(36, dtype=torch.long)
        # sample K style refs
        refs = random.sample(self.style_pool, min(self.K, len(self.style_pool)))
        ref_imgs = []
        for r in refs:
            ri = Image.open(r).convert('L').resize((80,80))
            arr = (np.array(ri).astype(np.float32)/127.5 - 1.0)
            ref_imgs.append(arr)
        ref_t = torch.tensor(np.stack(ref_imgs, axis=0)).unsqueeze(1)  # [K,1,H,W]
        return {'image': img_t, 'content': content, 'stroke': stroke, 'style_refs': ref_t}


def train_few_shot(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model backbone
    dp = DPFontModel(device=device).to(device)
    # if pretrained ckpt provided, load into dp (model.unet & t_emb)
    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        dp.load_state_dict(ckpt['model'], strict=False)
        print('Loaded pretrained checkpoint')

    # attr encoder but we'll use it for content+stroke fusion (style will come from style encoder)
    attr = AttributeEncoder().to(device)
    style_enc = StyleEncoder(out_dim=128).to(device)

    optimizer = torch.optim.AdamW(list(dp.unet.parameters()) + list(style_enc.parameters()) + list(attr.fusion_mlp.parameters()), lr=args.lr)

    # dataset
    ds = FewShotDataset(args.base_root, args.stroke_root, args.style_root, args.char_list, K=args.K, augment=args.augment)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # alphas for diffusion
    betas, alphas, alpha_bars = make_alphas(T=args.T, device=device)

    pinn = PINNLoss(D=0.1, sign_mode='laplace_only').to(device) if args.use_pinn else None

    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            x0 = batch['image'].to(device).float()
            content = batch['content'].to(device)
            stroke = batch['stroke'].to(device)
            style_refs = batch['style_refs'].to(device)  # [B,K,1,H,W]

            # aggregate style vector: encode each ref and average
            B,K,_,H,W = style_refs.shape
            style_refs_flat = style_refs.view(B*K,1,H,W)
            style_vecs = style_enc(style_refs_flat)
            style_vecs = style_vecs.view(B,K,-1).mean(dim=1)

            # content and stroke embeddings via attr's submodules
            c_embed = attr.content_embed(content)
            s_emb = attr.stroke_embed(stroke)
            s_emb = s_emb.view(B, -1)
            s_emb = attr.stroke_mlp(s_emb)

            # fuse
            z_in = torch.cat([c_embed, s_emb, style_vecs], dim=1)
            z = attr.fusion_mlp(z_in)

            # diffusion forward
            t = torch.randint(1, args.T, (x0.size(0),), device=device)
            eps = torch.randn_like(x0)
            sqrt_ab = torch.sqrt(alpha_bars[t])[:,None,None,None]
            sqrt_1_ab = torch.sqrt(1 - alpha_bars[t])[:,None,None,None]
            x_t = sqrt_ab * x0 + sqrt_1_ab * eps

            # predict
            t_emb = dp.t_emb(t)
            pred_eps = dp.unet(x_t, t_emb, z)
            loss_simple = F.mse_loss(pred_eps, eps)

            loss_pinn = torch.tensor(0.0, device=device)
            if pinn is not None:
                x0_pred = (x_t - sqrt_1_ab * pred_eps) / (torch.sqrt(alpha_bars[t])[:,None,None,None] + 1e-8)
                loss_pinn = pinn(x0_pred)

            loss = loss_simple + args.lambda_pinn * loss_pinn
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(dp.unet.parameters()) + list(style_enc.parameters()), 1.0)
            optimizer.step()

            if step % 50 == 0:
                print(f"step {step} loss_simple={loss_simple.item():.6f} loss_pinn={loss_pinn.item() if isinstance(loss_pinn, torch.Tensor) else loss_pinn:.6f}")
            step += 1

        # save per epoch
        outp = os.path.join(args.out, f'finetune_epoch_{epoch}.pt')
        os.makedirs(args.out, exist_ok=True)
        torch.save({'model': dp.state_dict(), 'style_enc': style_enc.state_dict(), 'attr_fusion': attr.fusion_mlp.state_dict()}, outp)
        print('Saved checkpoint', outp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_root', type=str, default=None, help='root with base images for content (multiple fonts)')
    parser.add_argument('--stroke_root', type=str, default='data/stroke_order')
    parser.add_argument('--style_root', type=str, default='data/style_refs')
    parser.add_argument('--char_list', type=str, default='20013,22269', help='comma separated unicode ints')
    parser.add_argument('--pretrained_ckpt', type=str, default='', help='path to pretrained ckpt')
    parser.add_argument('--out', type=str, default='out_finetune')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--lambda_pinn', type=float, default=0.01)
    parser.add_argument('--use_pinn', action='store_true')
    args = parser.parse_args()

    char_list = [int(x) for x in args.char_list.split(',')]
    args.char_list = char_list
    train_few_shot(args)
```
