"""
DP-Font 训练脚本
功能：实现DP-Font模型的训练流程，包括数据加载、模型训练、损失计算和模型保存
集成项：
 - AMP (混合精度训练)
 - EMA (模型权重指数移动平均)
 - checkpoint 保存包含 optimizer / scaler / ema_shadow
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda import amp

from attribute_encoder import AttributeEncoder
from dataset import DPFontDataset
from pinn import PINNLoss
from model.dp_model import DPFontModel
from utils import make_alphas, save_batch
from src.ema import EMA

# 参数解析器配置
parser = argparse.ArgumentParser(description='DP-Font 训练参数')
parser.add_argument('--data_root', type=str, default='data',
                   help='字体图像根目录，包含FontName/unicode.png结构')
parser.add_argument('--stroke_root', type=str, default='data/stroke_order',
                   help='笔画顺序数据目录，包含unicode.npy文件')
parser.add_argument('--fonts', type=str, default='FontA,FontB',
                   help='训练字体列表，逗号分隔')
parser.add_argument('--chars', type=str, default='20013',
                   help='训练字符unicode列表，逗号分隔')
parser.add_argument('--out', type=str, default='out',
                   help='输出目录，用于保存检查点和生成样本')
parser.add_argument('--epochs', type=int, default=1,
                   help='训练轮数')
parser.add_argument('--batch', type=int, default=8,
                   help='批次大小')
parser.add_argument('--lr', type=float, default=2e-5,
                   help='学习率')
parser.add_argument('--lambda_pinn', type=float, default=0.1,
                   help='PINN损失权重，控制物理约束强度')
parser.add_argument('--p_uncond', type=float, default=0.1,
                   help='无条件训练概率，用于分类器自由引导')
parser.add_argument('--save_every_steps', type=int, default=1000,
                   help='训练过程中每多少 step 保存一次样本')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
font_list = args.fonts.split(',')
char_list = [int(x) for x in args.chars.split(',')]

# 数据加载
dataset = DPFontDataset(os.path.join(args.data_root, 'images'), args.stroke_root, font_list, char_list)
dloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

# 模型与优化器
model = DPFontModel(device=device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# AMP scaler & EMA
scaler = amp.GradScaler()
ema = EMA(model, decay=0.9998)

# 扩散参数
T = 1000
betas, alphas, alpha_bars = make_alphas(T=T, device=device)

# PINN 损失（可以调整 D、sign_mode 等参数）
pinn = PINNLoss(D=0.1, sign_mode='plus').to(device)

# 训练循环
global_step = 0
for epoch in range(args.epochs):
    model.train()
    for batch in dloader:
        # 1. 准备数据
        x0 = batch['image'].to(device)  # 原始图像 [-1,1], shape [B,C,H,W]
        content = batch['content'].to(device)  # 字符内容ID (B,)
        stroke = batch['stroke'].to(device)  # 笔画顺序 (B,36)
        style = batch['style'].to(device)  # 字体样式ID (B,)

        B = x0.size(0)

        # 2. 扩散过程 - 随机采样时间步和噪声
        t = torch.randint(1, T, (B,), device=device)
        eps = torch.randn_like(x0, device=device)
        # 计算加噪系数
        sqrt_ab = torch.sqrt(alpha_bars[t])[:, None, None, None]   # shape [B,1,1,1]
        sqrt_1_ab = torch.sqrt(1 - alpha_bars[t])[:, None, None, None]
        # 加噪图像
        x_t = sqrt_ab * x0 + sqrt_1_ab * eps

        # 3. 分类器自由引导 - 随机使用无条件输入
        if torch.rand(1).item() < args.p_uncond:
            stroke_in = torch.zeros_like(stroke)
            style_in = torch.zeros_like(style)
        else:
            stroke_in = stroke
            style_in = style

        # 4. forward + 损失计算（使用 AMP 混合精度）
        with amp.autocast():
            # 模型直接返回预测噪声 eps_hat
            pred_eps = model.forward(x_t, t, content, stroke_in, style_in)

            # 简单 MSE 损失
            loss_simple = F.mse_loss(pred_eps, eps)

            # 5. 近似 x_{t-1} (用于 PINN 或调试)
            # alpha_t_vec: alphas[t] -> shape [B], alpha_bar_vec shape [B]
            alpha_t_vec = alphas[t]            # [B]
            alpha_bar_vec = alpha_bars[t]      # [B]
            # expand for image arithmetic where needed
            alpha_t_exp = alpha_t_vec[:, None, None, None]
            alpha_bar_exp = alpha_bar_vec[:, None, None, None]

            # 根据 DDPM 公式计算逆向均值 mu_theta (忽略随机项)
            mu = (1.0 / (torch.sqrt(alpha_t_exp) + 1e-12)) * (x_t - ((1.0 - alpha_t_exp) / (torch.sqrt(1.0 - alpha_bar_exp) + 1e-12)) * pred_eps)
            x_t_minus_1_approx = mu.detach()

            # 6. 预测原始图像 x0_pred（用于 PINN 损失或分析）
            sqrt_at = torch.sqrt(alpha_t_exp)
            sqrt_1_at = torch.sqrt(1 - alpha_bar_exp)
            x0_pred = (x_t - sqrt_1_at * pred_eps) / (sqrt_at + 1e-8)

            # 7. 计算 PINN 损失 (优先使用 x_t / pred_eps / alpha 信息的动态残差)
            # PINN 接口支持多种调用方式；这里传入 x_t + pred_eps + alpha 向量以启用动态残差计算
            try:
                loss_pinn = pinn(x_t, pred_eps=pred_eps, alpha_t=alpha_t_vec, alpha_bar_t=alpha_bar_vec)
            except TypeError:
                # 如果 PINN 实现较简单，回退到对 x0_pred 的拉普拉斯正则化
                loss_pinn = pinn(x0_pred)

            # 总损失
            loss = loss_simple + args.lambda_pinn * loss_pinn

        # 8. 反向传播与优化（AMP/GradScaler）
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # unscale before gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # EMA 更新（在 optimizer.step() 之后）
        ema.update()

        # 9. 日志记录
        if global_step % 100 == 0:
            # 尽量把 loss（tensor）转为 Python float
            l_simple = float(loss_simple.detach().cpu().item()) if isinstance(loss_simple, torch.Tensor) else float(loss_simple)
            l_pinn = float(loss_pinn.detach().cpu().item()) if isinstance(loss_pinn, torch.Tensor) else float(loss_pinn)
            l_tot = float(loss.detach().cpu().item())
            print(f"step {global_step}: L_simple={l_simple:.6f} L_pinn={l_pinn:.6f} total={l_tot:.6f}")

        # 10. 定期生成样本（使用 EMA 权重以获得更好采样）
        if global_step % args.save_every_steps == 0:
            model.eval()
            with torch.no_grad():
                # apply EMA shadow weights for sampling
                ema.apply_shadow()
                try:
                    sample_x = model.sample(
                        content[:min(8, B)].to(device),
                        stroke[:min(8, B)].to(device),
                        style[:min(8, B)].to(device),
                        T=100,
                        guidance_scale=3.0,
                        device=device
                    )
                finally:
                    # always restore original weights
                    ema.restore()

                # save sample batch (save_batch expects a tensor batch or list)
                save_batch(sample_x, args.out, prefix=f'step_{global_step}')
            model.train()

        global_step += 1

    # 每轮保存检查点，包含 optimizer / scaler / ema shadow
    ckpt_path = os.path.join(args.out, f'ckpt_epoch_{epoch}.pt')
    save_dict = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
        # 保存 scaler 状态，便于 resume 精确恢复 AMP 状态
        'scaler': scaler.state_dict(),
        # 保存 EMA 的 shadow dict，便于采样时恢复 EMA 权重
        'ema_shadow': ema.shadow
    }
    torch.save(save_dict, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

print("Training finished.")
