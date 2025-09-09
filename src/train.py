"""
DP-Font 训练脚本
功能：实现DP-Font模型的训练流程，包括数据加载、模型训练、损失计算和模型保存
"""
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
# 训练循环
global_step = 0
for epoch in range(args.epochs):
    for batch in dloader:
        # 1. 准备数据
        x0 = batch['image'].to(device)  # 原始图像 [-1,1]
        content = batch['content'].to(device)  # 字符内容ID
        stroke = batch['stroke'].to(device)  # 笔画顺序
        style = batch['style'].to(device)  # 字体样式ID

        # 2. 扩散过程 - 随机采样时间步和噪声
        t = torch.randint(1, T, (x0.size(0),), device=device)
        eps = torch.randn_like(x0)
        # 计算加噪系数
        sqrt_ab = torch.sqrt(alpha_bars[t])[:,None,None,None]
        sqrt_1_ab = torch.sqrt(1 - alpha_bars[t])[:,None,None,None]
        # 加噪图像
        x_t = sqrt_ab * x0 + sqrt_1_ab * eps

        # 3. 分类器自由引导 - 随机使用无条件输入
        if torch.rand(1).item() < args.p_uncond:
            stroke_in = torch.zeros_like(stroke)
            style_in = torch.zeros_like(style)
        else:
            stroke_in = stroke
            style_in = style

        # 4. 模型预测噪声
        pred_eps = model.forward(x_t, t, content, stroke_in, style_in)
        # 简单MSE损失
        loss_simple = F.mse_loss(pred_eps, eps)

        # 5. 近似x_{t-1} (用于后续计算)
        alpha_t = alphas[t][:,None,None,None]
        alpha_bar = alpha_bars[t][:,None,None,None]
        mu = (1/torch.sqrt(alpha_t)) * (x_t - ((1-alpha_t)/torch.sqrt(1-alpha_bar)) * pred_eps)
        x_t_minus_1_approx = mu.detach()

        # 6. 预测原始图像x0 (用于PINN损失)
        sqrt_at = torch.sqrt(alpha_t)
        sqrt_1_at = torch.sqrt(1 - alpha_bar)
        x0_pred = (x_t - sqrt_1_at * pred_eps) / (sqrt_at + 1e-8)

        # 7. 计算PINN损失 (物理约束)
        loss_pinn = pinn(x0_pred)

        # 8. 总损失 = 简单损失 + PINN损失权重 * PINN损失
        loss = loss_simple + args.lambda_pinn * loss_pinn

        # 9. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()

        # 10. 日志记录
        if global_step % 100 == 0:
            print(f"step {global_step}: L_simple={loss_simple.item():.6f} L_pinn={loss_pinn.item():.6f} total={loss.item():.6f}")

        # 11. 定期生成样本
        if global_step % 1000 == 0:
            with torch.no_grad():
                sample_x = model.sample(
                    content[:8].to(device), 
                    stroke[:8].to(device), 
                    style[:8].to(device), 
                    T=100, 
                    guidance_scale=3.0, 
                    device=device
                )
                save_batch(sample_x, args.out, prefix=f'step_{global_step}')

        global_step += 1

    # 每轮保存检查点
    torch.save(
        {
            'model': model.state_dict(),  # 模型参数
            'opt': optimizer.state_dict()  # 优化器状态
        }, 
        os.path.join(args.out, f'ckpt_epoch_{epoch}.pt')
    )

