"""
DP-Font 主模型模块
功能：实现基于扩散模型的字体生成系统，整合属性编码器、时间嵌入和UNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dp_unet import SimpleUNet, TimeEmbedding
from attribute_encoder import AttributeEncoder

class DPFontModel(nn.Module):
    """
    DP-Font 主模型，整合所有组件实现字体生成
    
    参数：
    - device: 计算设备 (默认'cuda')
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.attr = AttributeEncoder()  # 属性编码器
        self.t_emb = TimeEmbedding(128)  # 时间嵌入
        self.unet = SimpleUNet(         # UNet主干
            base_ch=32, 
            z_dim=512, 
            t_dim=128
        )
        self.device = device

    def forward(self, x_t, t, content_ids, stroke_orders, style_ids):
        """
        训练阶段前向传播，预测噪声
        
        参数：
        - x_t: 带噪声图像 [B,1,H,W]
        - t: 扩散时间步 [B]
        - content_ids: 字符内容ID [B]
        - stroke_orders: 笔画顺序 [B,36]
        - style_ids: 字体样式ID [B]
        
        返回：
        - pred_eps: 预测的噪声 [B,1,H,W]
        """
        # 编码属性
        z = self.attr(content_ids, stroke_orders, style_ids)  # [B,512]
        # 时间嵌入
        t_emb = self.t_emb(t)  # [B,128]
        # UNet预测噪声
        pred_eps = self.unet(x_t, t_emb, z)
        return pred_eps

    @torch.no_grad()
    def sample(self, content_ids, stroke_orders, style_ids, T=1000, guidance_scale=3.0, device='cuda'):
        """
        生成阶段采样，从噪声逐步生成字体图像
        
        参数：
        - content_ids: 字符内容ID [B]
        - stroke_orders: 笔画顺序 [B,36]
        - style_ids: 字体样式ID [B]
        - T: 扩散步数 (默认1000)
        - guidance_scale: 条件引导系数 (默认3.0)
        - device: 计算设备 (默认'cuda')
        
        返回：
        - x: 生成的字体图像 [B,1,80,80]
        """
        B = content_ids.size(0)
        # 初始化随机噪声
        x = torch.randn(B, 1, 80, 80, device=device)
        
        # 线性beta调度
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        # 反向扩散过程
        for i in reversed(range(T)):
            t = torch.tensor([i]*B, device=device)
            
            # 条件预测
            z_cond = self.attr(content_ids, stroke_orders, style_ids)
            t_emb = self.t_emb(t)
            eps_cond = self.unet(x, t_emb, z_cond)
            
            # 无条件预测 (零输入笔画和样式)
            z_uncond = self.attr(
                content_ids, 
                torch.zeros_like(stroke_orders), 
                torch.zeros_like(style_ids)
            )
            eps_uncond = self.unet(x, t_emb, z_uncond)
            
            # 引导式预测
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            # 计算均值
            alpha_t = alphas[i]
            alpha_bar = alpha_bars[i]
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar)
            mu = coef1 * (x - coef2 * eps)
            
            # 添加噪声 (最后一步不加)
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(betas[i])
                x = mu + sigma * noise
            else:
                x = mu
                
        return x
