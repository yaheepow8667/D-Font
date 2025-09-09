"""
指数移动平均(EMA)模块
功能：维护模型参数的指数移动平均值，用于模型权重平滑和稳定训练
"""

class EMA:
    def __init__(self, model, decay=0.9999):
        """
        初始化EMA
        
        参数：
        - model: 要跟踪的模型
        - decay: 衰减率 (默认0.9999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}  # 存储EMA权重
        self.backup = {}  # 备份原始权重
        
        # 初始化影子权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        更新EMA权重
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"参数 {name} 未在EMA影子权重中初始化"
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        """
        应用EMA权重到模型
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()  # 备份当前权重
                param.data.copy_(self.shadow[name])  # 应用EMA权重

    def restore(self):
        """
        恢复模型原始权重
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}  # 清空备份