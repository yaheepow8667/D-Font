"""
工具函数模块
功能：提供图像保存和扩散调度相关的实用函数
"""
import torch
import os
from torchvision.utils import save_image

def save_batch(img_tensor, out_dir, prefix='sample', nrow=8):
    """
    保存批量图像到指定目录
    
    参数：
    - img_tensor: 图像张量 (范围[-1,1]) [B,C,H,W]
    - out_dir: 输出目录路径
    - prefix: 文件名前缀 (默认'sample')
    - nrow: 每行显示的图像数量 (默认8)
    """
    os.makedirs(out_dir, exist_ok=True)  # 确保目录存在
    # 将图像从[-1,1]范围转换到[0,1]范围
    t = (img_tensor.clamp(-1,1) + 1.0) / 2.0
    # 保存图像网格
    save_image(t, os.path.join(out_dir, f"{prefix}.png"), nrow=nrow)

def make_alphas(T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
    """
    生成扩散过程的alpha调度参数
    
    参数：
    - T: 扩散步数 (默认1000)
    - beta_start: beta起始值 (默认1e-4)
    - beta_end: beta结束值 (默认0.02)
    - device: 计算设备 (默认'cpu')
    
    返回：
    - betas: beta值序列 [T]
    - alphas: alpha值序列 [T] (alpha = 1 - beta)
    - alpha_bars: alpha的累积乘积 [T]
    """
    # 线性beta调度
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # 计算累积乘积
    return betas, alphas, alpha_bars
