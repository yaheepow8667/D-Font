"""
UNet模型模块
功能：实现用于字体生成的简化UNet结构，支持条件嵌入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyResBlock(nn.Module):
    """
    小型残差块，支持条件缩放和偏移
    
    参数：
    - channels: 输入/输出通道数
    - emb_dim: 条件嵌入维度 (可选)
    """
    def __init__(self, channels, emb_dim=None):
        super().__init__()
        # 归一化层
        self.norm1 = nn.GroupNorm(8, channels)  # 8个组的分组归一化
        self.norm2 = nn.GroupNorm(8, channels)
        
        # 卷积层
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        # 条件投影MLP
        if emb_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),                     # 激活函数
                nn.Linear(emb_dim, channels*2)  # 输出缩放和偏移参数
            )
        else:
            self.mlp = None

    def forward(self, x, emb=None):
        """
        前向传播
        
        参数：
        - x: 输入特征 [B,C,H,W]
        - emb: 条件嵌入 [B,emb_dim] (可选)
        """
        h = self.norm1(x)
        h = F.silu(h)  # Sigmoid线性单元激活
        h = self.conv1(h)
        
        # 条件缩放和偏移
        if self.mlp is not None and emb is not None:
            scale_shift = self.mlp(emb)
            scale, shift = scale_shift.chunk(2, dim=1)
            # 调整维度以匹配特征图
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale) + shift  # 应用条件变换
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h  # 残差连接

class SimpleUNet(nn.Module):
    """
    简化版UNet模型，用于80x80灰度图像生成
    
    参数：
    - base_ch: 基础通道数 (默认64)
    - z_dim: 条件向量维度 (默认512)
    - t_dim: 时间嵌入维度 (默认128)
    """
    def __init__(self, base_ch=64, z_dim=512, t_dim=128):
        super().__init__()
        # 初始卷积
        self.init_conv = nn.Conv2d(1, base_ch, 3, padding=1)
        
        # 下采样路径
        self.down1 = nn.Sequential(
            TinyResBlock(base_ch, emb_dim=z_dim+t_dim),
            nn.AvgPool2d(2)  # 降采样
        )
        self.down2 = nn.Sequential(
            TinyResBlock(base_ch*2, emb_dim=z_dim+t_dim),
            nn.AvgPool2d(2)
        )
        
        # 中间层
        self.mid = TinyResBlock(base_ch*4, emb_dim=z_dim+t_dim)
        
        # 上采样路径
        self.up2 = nn.Sequential(TinyResBlock(base_ch*2, emb_dim=z_dim+t_dim))
        self.up1 = nn.Sequential(TinyResBlock(base_ch, emb_dim=z_dim+t_dim))
        
        # 最终卷积
        self.final = nn.Conv2d(base_ch, 1, 3, padding=1)
        
        # 通道调整层
        self.increase1 = nn.Conv2d(base_ch, base_ch*2, 1)  # 1x1卷积增加通道
        self.increase2 = nn.Conv2d(base_ch*2, base_ch*4, 1)
        self.decrease2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)  # 转置卷积上采样
        self.decrease1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)

    def forward(self, x, t_emb, z):
        """
        前向传播
        
        参数：
        - x: 输入图像 [B,1,80,80]
        - t_emb: 时间嵌入 [B,t_dim]
        - z: 条件向量 [B,z_dim]
        
        返回：
        - out: 生成图像 [B,1,80,80]
        """
        # 合并条件信息
        emb = torch.cat([z, t_emb], dim=1)  # [B, z_dim + t_dim]
        
        # 编码路径
        h0 = self.init_conv(x)      # [B,base_ch,80,80]
        h1 = self.down1[0](h0, emb) # 第一个残差块
        h1p = self.down1[1](h1)     # 下采样 [B,base_ch,40,40]
        h2 = self.increase1(h1p)    # 增加通道 [B,base_ch*2,40,40]
        h2 = self.down2[0](h2, emb)  # 第二个残差块
        h2p = self.down2[1](h2)     # 下采样 [B,base_ch*2,20,20]
        h3 = self.increase2(h2p)    # 增加通道 [B,base_ch*4,20,20]
        
        # 中间层
        h3 = self.mid(h3, emb)      # [B,base_ch*4,20,20]
        
        # 解码路径
        u2 = self.decrease2(h3)     # 上采样 [B,base_ch*2,40,40]
        u2 = self.up2[0](u2 + h2, emb)  # 残差连接+残差块
        u1 = self.decrease1(u2)     # 上采样 [B,base_ch,80,80]
        u1 = self.up1[0](u1 + h1, emb)  # 残差连接+残差块
        
        # 最终输出
        out = self.final(u1)        # [B,1,80,80]
        return out

class TimeEmbedding(nn.Module):
    """
    时间嵌入模块，将离散时间步转换为连续向量
    
    参数：
    - dim: 输出嵌入维度
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 投影网络
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),  # 线性变换
            nn.SiLU(),           # 激活函数
            nn.Linear(dim, dim)  # 二次变换
        )

    def forward(self, t):
        """
        前向传播
        
        参数：
        - t: 时间步 [B] (整数张量)
        
        返回：
        - emb: 时间嵌入 [B,dim]
        """
        half = self.dim // 2
        # 计算频率因子
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0)) * 
            torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        
        # 计算位置参数
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        
        # 正弦余弦编码
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        
        # 处理奇数维度
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.size(0),1, device=t.device)], dim=1)
            
        return self.proj(emb)  # 非线性变换