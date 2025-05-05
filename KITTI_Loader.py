import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class StereoDataset(Dataset):
    def __init__(self, left_path, right_path, disp_path, max_disp=192):
        self.left_path = left_path
        self.right_path = right_path
        self.disp_path = disp_path
        self.max_disp = max_disp
        self.file_list = sorted([f for f in os.listdir(left_path) if f.endswith(".png")])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        left_filename = self.file_list[idx]
        base_name_left = os.path.splitext(left_filename)[0]

        # 生成对应的右视图文件名
        if not base_name_left.endswith("_10"):
            raise ValueError(f"左视图文件名格式异常: {left_filename}")
        base_name_right = base_name_left.replace("_10", "_11")

        # 路径校验
        left_path = os.path.join(self.left_path, f"{base_name_left}.png")
        right_path = os.path.join(self.right_path, f"{base_name_right}.png")
        disp_path = os.path.join(self.disp_path, f"{base_name_left}.png")

        # 读取图像数据
        left_img = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)
        disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # 视差数据处理
        disp = disp / 256.0  # KITTI数据集视差存储格式转换
        disp[disp > self.max_disp] = 0
        disp[disp < 0] = 0

        # 统一调整尺寸
        HEIGHT, WIDTH = 375, 1242
        left_img = cv2.resize(left_img, (WIDTH, HEIGHT))
        right_img = cv2.resize(right_img, (WIDTH, HEIGHT))
        disp = cv2.resize(disp, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

        # 转换为Tensor
        left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).float() / 255.0
        right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).float() / 255.0
        disp_tensor = torch.from_numpy(disp).float().unsqueeze(0)

        return left_tensor, right_tensor, disp_tensor