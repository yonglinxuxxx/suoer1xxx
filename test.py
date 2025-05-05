import torch
import numpy as np
from torch.utils.data import DataLoader
from model import OptimizedGCNet, get_gcnet  # 请确保模型代码在model.py中
from dataset import StereoDataset  # 请确保数据集代码在dataset.py中
import cv2
import os

def main():
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_disp = 128  # 必须与训练时的设置一致
    batch_size = 2

    # 初始化模型
    model = get_gcnet(height=375, width=1242, max_disp=max_disp).to(device)
    checkpoint_path = "./checkpoints/gcnet_best_202504301155.pth"  # 指定训练好的权重路径
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    output_dir = "./disparity_output"
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据集
    dataset = StereoDataset(
        left_path="./Dataset/left",
        right_path="./Dataset/right",
        disp_path="./Dataset/disp",
        max_disp=max_disp
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化评估指标
    total_error = 0.0
    total_outliers = 0
    total_valid_pixels = 0

    with torch.no_grad():
        for batch_idx, (left, right, disp_gt) in enumerate(dataloader):
            left, right, disp_gt = left.to(device), right.to(device), disp_gt.to(device)
            disp_pred = model(left, right)

            # 检查输出范围和形状
            print(
                f"Batch {batch_idx}: Disp range [{disp_pred.min().item()}, {disp_pred.max().item()}], Shape: {disp_pred.shape}")

            disp_pred_np = disp_pred.cpu().numpy()
            for i in range(disp_pred_np.shape[0]):
                filename = f"disp_{batch_idx}_{i}.png"
                filepath = os.path.join(output_dir, filename)
                disparity_uint16 = (disp_pred_np[i].squeeze() * 256).astype(np.uint16)
                cv2.imwrite(filepath, disparity_uint16)

            # 创建有效像素掩码
            valid_mask = (disp_gt > 0) & (disp_gt <= max_disp)
            valid_pixels = valid_mask.sum().item()

            if valid_pixels == 0:
                continue

            # 计算绝对误差
            abs_error = torch.abs(disp_pred - disp_gt)
            abs_error[~valid_mask] = 0  # 忽略无效像素

            # 统计指标
            total_error += abs_error.sum().item()
            total_outliers += (abs_error[valid_mask] > 3).sum().item()
            total_valid_pixels += valid_pixels

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")

    # 计算最终指标
    avg_noc = total_error / total_valid_pixels
    out_noc = (total_outliers / total_valid_pixels) * 100

    print("\nEvaluation Results:")
    print(f"Avg-Noc: {avg_noc:.4f} pixels")
    print(f"Out-Noc: {out_noc:.2f}%")
    print(f"Total valid pixels: {total_valid_pixels}")


if __name__ == "__main__":
    main()