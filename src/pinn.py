"""
物理信息神经网络(PINN)损失模块
功能：计算基于物理约束的损失，增强生成图像的物理合理性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNLoss(nn.Module):
    def __init__(self, D=0.1, loss_type='mse', sign_mode='plus', learnable_D=False, clip_lap=10.0):
        """
        改进版的DP-Font PINN损失函数
        
        参数：
        - D: 扩散系数 (默认0.1)
        - loss_type: 损失类型，'mse'或'l1' (默认'mse')
        - sign_mode: 符号模式：
            'plus' 使用残差 = du_dt + D * laplacian (推荐)
            'minus' 使用残差 = du_dt - D * laplacian
            'laplace_only' 仅使用laplacian(u)
        - learnable_D: 是否将D作为可学习参数 (默认False)
        - clip_lap: 拉普拉斯值裁剪阈值，避免数值问题 (默认10.0)
        """
        super().__init__()
        self.loss_type = loss_type
        self.sign_mode = sign_mode
        self.clip_lap = float(clip_lap)
        if learnable_D:
            self.D = nn.Parameter(torch.tensor(float(D)))  # 可学习的扩散系数
        else:
            # 注册为buffer以便设备转移和保存
            self.register_buffer('D', torch.tensor(float(D)))
        
        # 定义3x3拉普拉斯算子卷积核
        kernel = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kernel', kernel)

    def compute_lap(self, u):
        """
        计算图像的拉普拉斯算子
        
        参数：
        - u: 输入图像张量 [B,C,H,W]
        
        返回：
        - lap: 拉普拉斯计算结果 [B,C,H,W]
        """
        B,C,H,W = u.shape
        k = self.kernel.repeat(C,1,1,1)  # 为每个通道复制核 [C,1,3,3]
        u_pad = F.pad(u, (1,1,1,1), mode='reflect')  # 反射填充
        lap = F.conv2d(u_pad, k, groups=C)  # 分组卷积
        
        # 数值稳定性处理：裁剪拉普拉斯值
        if self.clip_lap is not None and self.clip_lap > 0.0:
            lap = torch.clamp(lap, -self.clip_lap, self.clip_lap)
        return lap

    def forward(self, x_t, pred_eps=None, x_t_minus_1=None, alpha_t=None, alpha_bar_t=None, delta_t=1.0):
        """
        计算PINN损失
        
        推荐用法：提供x_t和pred_eps以及alpha_t和alpha_bar_t，以便近似x_{t-1}
        如果显式提供了x_t_minus_1，则会直接使用
        
        参数：
        - x_t: 当前含噪图像 [B,C,H,W]
        - pred_eps: UNet预测的噪声 [B,C,H,W] (可选)
        - x_t_minus_1: 预先计算的前一时刻图像 [B,C,H,W] (可选)
        - alpha_t, alpha_bar_t: 时间t对应的alpha值 (使用pred_eps时必需)
        - delta_t: 时间步长 (默认1.0)
        
        返回：
        - 标量损失值 (基于MSE或L1的残差损失)
        """
        device = x_t.device
        
        # 计算时间导数du/dt
        if x_t_minus_1 is None:
            if pred_eps is None or alpha_t is None or alpha_bar_t is None:
                # 回退方案：仅在x_t上计算拉普拉斯
                lap = self.compute_lap(x_t)
                resid = lap if self.sign_mode == 'laplace_only' else (self.D * lap)
            else:
                # 计算x_{t-1}的近似值mu_theta
                sqrt_alpha_t = torch.sqrt(alpha_t).view(-1,1,1,1)
                sqrt_1_alpha_bar = torch.sqrt(1.0 - alpha_bar_t).view(-1,1,1,1)
                coef1 = 1.0 / (sqrt_alpha_t + 1e-12)  # 避免除零
                coef2 = ((1.0 - alpha_t) / (sqrt_1_alpha_bar + 1e-12)).view(-1,1,1,1)
                mu = coef1 * (x_t - coef2 * pred_eps)
                x_t_minus_1 = mu.detach()  # 分离计算图
                du_dt = (x_t - x_t_minus_1) / float(delta_t)
                
                # 计算拉普拉斯和残差
                lap = self.compute_lap(x_t)
                if self.sign_mode == 'minus':
                    resid = du_dt - self.D * lap  # 热扩散方程形式
                elif self.sign_mode == 'plus':
                    resid = du_dt + self.D * lap  # 反向扩散形式
                else:
                    resid = du_dt + self.D * lap  # 默认使用正向扩散
        else:
            # 使用显式提供的x_t_minus_1
            du_dt = (x_t - x_t_minus_1) / float(delta_t)
            lap = self.compute_lap(x_t)
            if self.sign_mode == 'minus':
                resid = du_dt - self.D * lap
            elif self.sign_mode == 'plus':
                resid = du_dt + self.D * lap
            else:
                resid = du_dt + self.D * lap

        # 返回MSE或L1损失
        if self.loss_type == 'mse':
            return F.mse_loss(resid, torch.zeros_like(resid))
        else:
            return F.l1_loss(resid, torch.zeros_like(resid))