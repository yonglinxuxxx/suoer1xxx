import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from pathlib import Path


class StereoDataset(Dataset):
    def __init__(self, left_path="./Dataset/left", right_path="./Dataset/right", disp_path="./Dataset/disp", max_disp=192, debug_mode=False):
        """
        参数说明：
        - debug_mode: 启用调试模式时会显示加载样本的信息
        """

        # 路径验证
        self.left_path = Path(left_path)
        self.right_path = Path(right_path)
        self.disp_path = Path(disp_path)

        if not (self.left_path.exists() and self.right_path.exists() and self.disp_path.exists()):
            raise FileNotFoundError("一个或多个数据目录不存在")

        # 获取文件列表并验证配对
        self.file_list = sorted([f for f in os.listdir(self.left_path) if f.endswith(".png")])
        self._validate_file_pairs()

        self.max_disp = max_disp
        self.debug_mode = debug_mode
        self.target_size = (1242, 375)  # (width, height) KITTI原始尺寸

    def _validate_file_pairs(self):
        """验证左右视图文件是否成对存在"""
        missing_pairs = []
        for left_file in self.file_list:
            base_name = left_file.split('_')[0]
            right_file = f"{base_name}_11.png"
            if not (self.right_path / right_file).exists():
                missing_pairs.append((left_file, right_file))

        if missing_pairs:
            error_msg = f"发现{len(missing_pairs)}个未配对文件，例如：{missing_pairs[:3]}"
            raise FileNotFoundError(error_msg)

    def __len__(self):
        return len(self.file_list)

    def _preprocess_disp(self, disp):
        """视差图预处理流程（修正版）"""
        disp = cv2.resize(disp, self.target_size, interpolation=cv2.INTER_NEAREST)
        disp = disp.astype(np.float32)
        disp = disp / 256.0  # 关键修正：KITTI视差数据转换
        disp[disp > self.max_disp] = 0
        disp[disp < 0] = 0
        return torch.from_numpy(disp).unsqueeze(0)

    def _preprocess_image(self, img):
        """图像预处理流程"""
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

    def __getitem__(self, idx):
        try:
            # 构造文件路径
            left_filename = self.file_list[idx]
            base_name = left_filename.split('_')[0]

            left_path = self.left_path / left_filename
            right_path = self.right_path / f"{base_name}_11.png"
            disp_path = self.disp_path / left_filename

            # 读取图像数据
            left_img = cv2.cvtColor(cv2.imread(str(left_path)), cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(cv2.imread(str(right_path)), cv2.COLOR_BGR2RGB)
            disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)

            # 调试信息
            if self.debug_mode:
                print(f"样本 {idx}:")
                print(f"左图尺寸: {left_img.shape}, 右图尺寸: {right_img.shape}")
                print(f"视差范围: {np.min(disp)}-{np.max(disp)} (预处理前)")

            # 预处理
            left_tensor = self._preprocess_image(left_img)
            right_tensor = self._preprocess_image(right_img)
            disp_tensor = self._preprocess_disp(disp)

            # 后处理验证
            if disp_tensor.max() > self.max_disp:
                raise ValueError(f"视差值超过最大设定值 {self.max_disp}")

            return left_tensor, right_tensor, disp_tensor

        except Exception as e:
            print(f"加载样本 {idx} 时发生错误: {str(e)}")
            return None  # 或执行其他错误处理策略


# 使用示例
if __name__ == "__main__":
    def test_dataset():
        dataset = StereoDataset(debug_mode=True)
        print(f"数据集大小: {len(dataset)} 个样本")

        # 测试数据加载
        sample = dataset[0]
        if sample is not None:
            left, right, disp = sample
            print("\n预处理后张量形状:")
            print(f"左图: {left.shape} (dtype: {left.dtype})")
            print(f"右图: {right.shape} (dtype: {right.dtype})")
            print(f"视差: {disp.shape} (dtype: {disp.dtype}, 范围: {disp.min()}-{disp.max()})")

        # 创建DataLoader测试
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        for batch in dataloader:
            if None in batch:  # 处理可能的错误样本
                continue
            left, right, disp = batch
            print(f"\n批次数据形状:")
            print(f"左图: {left.shape}")
            print(f"右图: {right.shape}")
            print(f"视差: {disp.shape}")
            break


    test_dataset()