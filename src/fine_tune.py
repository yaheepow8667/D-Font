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
