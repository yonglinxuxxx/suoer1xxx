# draw.py
import torch
from torchviz import make_dot
from model import get_gcnet  # 假设模型代码保存在model.py中


def visualize_feature_extract():
    # 创建模型并获取特征提取模块
    model = get_gcnet()
    feat_extract = model.feature_extract

    # 生成随机输入
    x = torch.randn(1, 3, 256, 512)

    # 前向传播获取计算图
    out = feat_extract(x)

    # 使用torchviz生成可视化图（仅显示顶层模块）
    dot = make_dot(out,
                   params=dict(feat_extract.named_parameters()),
                   show_attrs=True,
                   show_saved=True)

    # 保存为PDF并渲染
    dot.render("feature_extract_architecture", format="png", cleanup=True)
    print("架构图已保存为 feature_extract_architecture.png")


if __name__ == "__main__":
    visualize_feature_extract()