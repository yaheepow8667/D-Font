"""
字体数据集加载模块
功能：加载字体图像和对应的笔画顺序数据，为DP-Font模型提供训练数据
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class DPFontDataset(Dataset):
    """
    字体数据集类，继承自torch.utils.data.Dataset
    
    参数：
    - image_root: 字体图像根目录，结构为 FontName/unicode.png
    - stroke_root: 笔画顺序数据根目录，包含unicode.npy文件
    - font_list: 使用的字体名称列表
    - char_list: 使用的字符unicode码列表
    - transform: 可选的数据增强变换
    """
    def __init__(self, image_root, stroke_root, font_list, char_list, transform=None):
        self.image_root = image_root  # 字体图像根目录
        self.stroke_root = stroke_root  # 笔画顺序数据根目录
        self.font_list = font_list  # 字体名称列表
        self.char_list = char_list  # 字符unicode码列表
        self.transform = transform  # 数据增强变换

        # 构建样本列表：(图像路径, 笔画路径, 字体名称, 字符代码)
        self.samples = []
        for font in font_list:
            for char_code in char_list:
                img_path = os.path.join(image_root, font, f"{char_code:05d}.png")
                stroke_path = os.path.join(stroke_root, f"{char_code:05d}.npy")
                if os.path.exists(img_path) and os.path.exists(stroke_path):
                    self.samples.append((img_path, stroke_path, font, char_code))

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回：
        - dict: 包含以下键值：
            - image: 归一化到[-1,1]的图像张量 (1,80,80)
            - stroke: 笔画顺序张量 (36,)
            - style: 字体样式ID (long)
            - content: 字符内容ID (long)
        """
        img_path, stroke_path, font_name, char_code = self.samples[idx]
        
        # 加载并预处理图像
        img = Image.open(img_path).convert('L').resize((80,80))  # 转为灰度并调整大小
        arr = np.array(img).astype(np.float32)
        # 归一化到[-1,1]范围
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0
        
        # 加载笔画顺序数据
        stroke = torch.from_numpy(np.load(stroke_path)).long()  # (36,)长整型张量
        
        # 生成样式和内容ID
        style_id = torch.tensor(abs(hash(font_name)) % 200, dtype=torch.long)
        content_id = torch.tensor(char_code, dtype=torch.long)
        
        return {
            'image': tensor,      # 图像张量
            'stroke': stroke,     # 笔画顺序
            'style': style_id,    # 字体样式ID
            'content': content_id # 字符内容ID
        }