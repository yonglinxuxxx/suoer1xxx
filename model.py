import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio=6):
        super().__init__()
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.expand_ratio = expand_ratio
        self.use_res_connect = self.stride == 1 and in_ch == out_ch

        hidden_dim = in_ch * expand_ratio
        layers = []

        if expand_ratio != 1:
            # 扩展层
            layers.append(nn.Conv2d(in_ch, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # 深度卷积
        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        )
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # 投影层
        layers.append(nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def get_InvertedResidual():
    return InvertedResidual()

class CostVolume(nn.Module):
    def __init__(self, max_disp):
        super().__init__()
        self.max_disp = max_disp

    def forward(self, left, right):
        B, C, H, W = left.shape
        cost_vol = []
        for d in range(self.max_disp):
            if d == 0:
                left_slice = left
                right_slice = right
            else:
                left_slice = left[..., d:]
                right_slice = right[..., :-d]
                left_slice = F.pad(left_slice, (0, d))
                right_slice = F.pad(right_slice, (d, 0))
            cost = torch.cat([left_slice, right_slice], dim=1)
            cost_vol.append(cost.unsqueeze(2))
        return torch.cat(cost_vol, dim=2)

class OptimizedGCNet(nn.Module):
    def __init__(self, height=375, width=1242, max_disp=128):
        super().__init__()
        self.max_disp = max_disp // 4
        self.cost_volume = CostVolume(self.max_disp)
        self.target_height = height
        self.target_width = width

        # 特征提取网络（MobileNetV2风格）
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),  # H/2
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            InvertedResidual(16, 24, stride=2, expand_ratio=6),  # H/4
            InvertedResidual(24, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
        )

        # 3D卷积部分（保持不变）
        self.conv3d = nn.Sequential(
            nn.Conv3d(64, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 16, 3, 1, 1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        # 上采样部分（保持不变）
        self.up_sample = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3,
                               stride=(1, 2, 2),
                               padding=1,
                               output_padding=(0, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        # 亚像素上采样（保持不变）
        self.subpixel = nn.Sequential(
            nn.Conv2d(8 * self.max_disp, 32, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1)
        )

    def forward(self, left, right):
        # 特征提取（MobileNetV2）
        feat_l = self.feature_extract(left)  # [B,32,H/4,W/4]
        feat_r = self.feature_extract(right)

        # 后续处理与原始保持一致
        cost = self.cost_volume(feat_l, feat_r)
        cost = self.conv3d(cost)
        up = self.up_sample(cost)

        B, C, D, H, W = up.shape
        disp = up.view(B, -1, H, W)
        disp = self.subpixel(disp)
        disp = F.interpolate(disp,
                             size=(self.target_height, self.target_width),
                             mode='bilinear',
                             align_corners=True)
        return torch.sigmoid(disp) * self.max_disp * 4

def get_gcnet(height=256, width=512, max_disp=64):
    return OptimizedGCNet(height, width, max_disp)

#if __name__ == "__main__":
    #model = get_gcnet()
    #left = torch.randn(2, 3, 256, 512)
    #right = torch.randn(2, 3, 256, 512)
    #disp = model(left, right)
    #print(disp.shape)  # 应该输出 torch.Size([2, 1, 256, 512])
