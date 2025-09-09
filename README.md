# DP-Font 复现项目(最小实现)

## 项目概述
DP-Font是一个基于扩散概率模型的字体生成系统，能够根据字符内容和笔画顺序生成不同风格的字体。

**核心功能**:
- 基于扩散模型的字体生成
- 支持多种字体风格的条件生成
- 结合笔画顺序信息(PINN)提升生成质量
- 集成的训练和采样流程

**系统架构**:
1. 属性编码器: 编码字符内容、笔画顺序和风格
2. 时间嵌入层: 处理扩散时间步
3. UNet主干网络: 包含下采样和上采样块的核心生成网络

**数据实体**:
- 字体图像: `data/images/<字体名称>/<unicode>.png`
- 笔画顺序: `data/stroke_order/<unicode>.npy` (36维向量)
- 模型检查点: 保存的训练状态

**工作流程**:
1. 数据准备:
   - 整理字体图像和笔画顺序数据
2. 训练:
   - 加载数据集
   - 训练扩散模型(含PINN损失)
   - 保存模型检查点
3. 生成:
   - 加载训练好的模型
   - 根据输入条件生成字体图像
   - 保存生成结果

本项目提供了一个基于小型UNet的最小可运行DP-Font实现框架，适合作为工程开发和实验的起点。

项目结构:
- src/: 核心Python模块
- data/: 存放图像和笔画顺序numpy文件

快速开始(示例):
1. 准备数据:
   - data/images/<字体名称>/<unicode:05d>.png
   - data/stroke_order/<unicode:05d>.npy  # 形状(36,)
2. 安装依赖:
   pip install -r requirements.txt
3. 运行训练(示例):
   python src/train.py --data_root data --stroke_root data/stroke_order --fonts FontA,FontB --chars 20013 --epochs 1 --batch 8

这是一个最小实现框架。完整设计说明、PINN变体和训练技巧请参考相关讨论。

# 包含演示数据集
本项目在`data/`目录下包含一个小型演示数据集，包含两种字体(FontA, FontB)和两个字符(编码点20013, 22269)，用于训练脚本的简单测试。
