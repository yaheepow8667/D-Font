"""
物理信息神经网络(PINN)损失模块
功能：计算基于物理约束的损失，增强生成图像的物理合理性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNLoss(nn.Module):
    """
    PINN损失函数，基于热扩散方程约束
    
    参数：
    - D: 扩散系数 (默认0.1)
    - loss_type: 损失类型，'mse'或'l1' (默认'mse')
    - sign_mode: 符号模式：
        'plus': resid = du_dt + D * laplacian(u)
        'minus': resid = du_dt - D * laplacian(u)
        'laplace_only': 仅使用laplacian(u)
    """
    def __init__(self, D=0.1, loss_type='mse', sign_mode='plus'):
        super().__init__()
        self.D = D            # 扩散系数
        self.loss_type = loss_type  # 损失类型
        self.sign_mode = sign_mode  # 符号模式
        
        # 拉普拉斯算子卷积核 (3x3)
        kernel = torch.tensor([
            [0., 1., 0.],
            [1.,-4., 1.],
            [0., 1., 0.]
        ], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kernel', kernel)  # 注册为不参与训练的buffer

    def compute_lap(self, u):
        """
        计算图像的拉普拉斯算子
        
        参数：
        - u: 输入图像张量 [B,C,H,W]
        
        返回：
        - lap: 拉普拉斯计算结果 [B,C,H,W]
        """
        B,C,H,W = u.shape
        k = self.kernel.repeat(C,1,1,1)  # 为每个通道复制核
        u_pad = F.pad(u, (1,1,1,1), mode='reflect')  # 反射填充
        lap = F.conv2d(u_pad, k, groups=C)  # 分组卷积
        return lap

    def forward(self, u_t, u_t_minus_1=None):
        """
        前向传播计算PINN损失
        
        参数：
        - u_t: 当前时刻图像 [B,C,H,W]
        - u_t_minus_1: 前一时刻图像 [B,C,H,W] (可选)
        
        返回：
        - loss: 物理约束损失值
        """
        # 计算拉普拉斯项
        lap = self.compute_lap(u_t)
        
        if self.sign_mode == 'laplace_only' or u_t_minus_1 is None:
            # 仅拉普拉斯模式
            resid = lap
        else:
            # 计算时间导数
            du_dt = (u_t - u_t_minus_1)
            
            # 根据符号模式组合项
            if self.sign_mode == 'minus':
                resid = du_dt - self.D * lap  # 热扩散方程
            else:
                resid = du_dt + self.D * lap  # 反向扩散
            
        # 计算损失
        if self.loss_type == 'mse':
            return F.mse_loss(resid, torch.zeros_like(resid))
        else:
            return F.l1_loss(resid, torch.zeros_like(resid))
