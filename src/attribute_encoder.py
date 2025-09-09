"""
属性编码器模块
功能：将字符内容、笔画顺序和字体样式编码为联合特征表示
"""
import torch
import torch.nn as nn

class AttributeEncoder(nn.Module):
    """
    属性编码器，将三种属性编码为联合特征
    
    参数：
    - content_dim: 字符内容编码维度 (默认128)
    - stroke_dim: 笔画顺序编码维度 (默认128)
    - style_dim: 字体样式编码维度 (默认128)
    - output_dim: 输出特征维度 (默认512)
    - max_strokes: 最大笔画数 (默认36)
    - num_chars: 字符数量 (默认20902)
    - num_styles: 样式数量 (默认200)
    """
    def __init__(self, content_dim=128, stroke_dim=128, style_dim=128,
                 output_dim=512, max_strokes=36, num_chars=20902, num_styles=200):
        super().__init__()
        # 字符内容嵌入层 (unicode -> content_dim)
        self.content_embed = nn.Embedding(num_chars, content_dim)
        
        # 笔画顺序嵌入层 (6种笔画类型 -> 16维)
        self.stroke_embed = nn.Embedding(6, 16)  # 0-5映射
        
        # 笔画顺序多层感知机 (max_strokes*16 -> stroke_dim)
        self.stroke_mlp = nn.Sequential(
            nn.Linear(max_strokes * 16, 256),  # 展开后全连接
            nn.ReLU(),                        # 激活函数
            nn.Linear(256, stroke_dim)        # 降维到stroke_dim
        )
        
        # 字体样式嵌入层 (style_id -> style_dim)
        self.style_embed = nn.Embedding(num_styles, style_dim)
        
        # 特征融合多层感知机
        self.fusion_mlp = nn.Sequential(
            nn.Linear(content_dim + stroke_dim + style_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)  # 二次变换增强表达能力
        )

    def forward(self, content_ids, stroke_orders, style_ids):
        """
        前向传播
        
        参数：
        - content_ids: 字符内容ID [batch_size]
        - stroke_orders: 笔画顺序 [batch_size, 36]
        - style_ids: 字体样式ID [batch_size]
        
        返回：
        - z: 联合特征表示 [batch_size, output_dim]
        """
        # 字符内容编码
        c = self.content_embed(content_ids)  # [B, content_dim]
        
        # 笔画顺序编码
        s = self.stroke_embed(stroke_orders)  # [B, 36, 16]
        s = s.view(s.size(0), -1)             # 展平 [B, 36*16]
        s = self.stroke_mlp(s)               # [B, stroke_dim]
        
        # 字体样式编码
        sty = self.style_embed(style_ids)     # [B, style_dim]
        
        # 特征融合
        z = self.fusion_mlp(torch.cat([c, s, sty], dim=1))  # [B, output_dim]
        return z