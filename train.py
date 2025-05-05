import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import os
import torch.nn as nn
from model import OptimizedGCNet
from KITTI_Loader import StereoDataset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt  # 新增导入
import numpy as np

# 训练配置
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "./Dataset"
    checkpoint_dir = "./checkpoints"
    plot_dir = "./plots"  # 新增：保存曲线图的目录
    batch_size = 2
    max_disp = 128
    grad_accum = 2
    use_amp = True
    img_height = 375
    img_width = 1242
    learning_rate = 0.001
    num_epochs = 50

def main():
    cfg = Config()
    scaler = GradScaler()

    # 创建目录
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.plot_dir, exist_ok=True)  # 新增：创建plot目录

    # 模型初始化
    model = OptimizedGCNet(
        height=cfg.img_height,
        width=cfg.img_width,
        max_disp=cfg.max_disp
    ).to(cfg.device)

    # 数据加载
    train_dataset = StereoDataset(
        os.path.join(cfg.data_root, "left"),
        os.path.join(cfg.data_root, "right"),
        os.path.join(cfg.data_root, "disp"),
        cfg.max_disp  # 确保max_disp一致
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.SmoothL1Loss()

    best_loss = float('inf')
    loss_history = []  # 新增：记录损失
    mae_history = []   # 新增：记录MAE

    # 训练循环
    for epoch in range(cfg.num_epochs):
        current_epoch = epoch + 1  # 转换为1-based计数
        if current_epoch > 30:
            new_lr = 0.00025
        elif current_epoch > 20:
            new_lr = 0.0005
        else:
            new_lr = cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        model.train()
        running_loss = 0.0
        running_mae = 0.0  # 新增：累计MAE
        total_samples = 0

        for batch_idx, (left, right, disp_gt) in enumerate(train_loader):
            left = left.to(cfg.device)
            right = right.to(cfg.device)
            disp_gt = disp_gt.to(cfg.device)  # 修正：移除 * cfg.max_disp

            # 前向传播
            with autocast(enabled=cfg.use_amp):
                pred_disp = model(left, right)
                loss = criterion(pred_disp, disp_gt)

            # 计算MAE（新增）
            with torch.no_grad():
                mae = torch.mean(torch.abs(pred_disp - disp_gt)).item()
                running_mae += mae * left.size(0)
                running_loss += loss.item() * left.size(0)
                total_samples += left.size(0)

            # 反向传播
            scaler.scale(loss).backward()
            if (batch_idx + 1) % cfg.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # 计算epoch平均损失和MAE
        epoch_loss = running_loss / total_samples
        epoch_mae = running_mae / total_samples
        loss_history.append(epoch_loss)
        mae_history.append(epoch_mae)
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}], Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f}")

        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_name = f"gcnet_best_{datetime.now().strftime('%Y%m%d%H%M')}.pth"
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, model_name))

        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(cfg.checkpoint_dir, checkpoint_name))

    # 在训练循环之后添加以下改进的绘图代码
    plt.figure(figsize=(12, 6), dpi=300)  # 增大画布尺寸和分辨率

    # 计算动态Y轴范围
    combined_values = loss_history + mae_history
    y_min = min(combined_values)
    y_max = max(combined_values)
    padding = (y_max - y_min) * 0.15  # 15%的留白空间
    y_lower = max(y_min - padding, 0)  # 确保最小值不低于0
    y_upper = y_max + padding

    # 绘制曲线
    epochs = range(1, len(loss_history) + 1)
    plt.plot(epochs, loss_history, 'o-', linewidth=1.5, markersize=6, label='Training Loss')
    plt.plot(epochs, mae_history, 's-', linewidth=1.5, markersize=6, label='Training MAE')

    # 坐标轴设置
    plt.xticks(epochs, fontsize=10)  # 每个epoch显示刻度
    plt.yticks(np.linspace(y_lower, y_upper, 10), fontsize=10)  # 更密集的Y轴刻度
    plt.xlim(0.5, len(loss_history) + 0.5)  # 留出左右边距
    plt.ylim(y_lower, y_upper)

    # 网格和标签
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=12, labelpad=8)
    plt.ylabel('Value', fontsize=12, labelpad=8)
    plt.title('Training Metrics with Refined Scale', fontsize=14, pad=15)

    # 添加数值标注（每两个epoch标注一次）
    for i, (loss, mae) in enumerate(zip(loss_history, mae_history)):
        if (i + 1) % 2 == 0 or (i + 1) == len(loss_history):
            plt.text(i + 1, loss + 0.005, f'{loss:.3f}', ha='center', fontsize=8, color='tab:blue')
            plt.text(i + 1, mae - 0.005, f'{mae:.3f}', ha='center', fontsize=8, color='tab:orange', va='top')

    # 图例和样式调整
    plt.legend(fontsize=10, loc='upper right', framealpha=0.9)
    plt.tight_layout()  # 自动调整布局

    # 保存改进后的图表
    plt.savefig(os.path.join(cfg.plot_dir, 'refined_training_curves.png'),
                bbox_inches='tight',
                facecolor='white')
    plt.close()

    print("✅ 训练完成！")

if __name__ == "__main__":
    main()