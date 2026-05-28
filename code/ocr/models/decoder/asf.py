'''
*****************************************************************************************
* DBNet++ - Adaptive Scale Fusion (ASF) Module
*
* 참고 논문:
* Real-time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion
* https://arxiv.org/pdf/2202.10304.pdf
*
* ASF는 두 단계로 구성:
*   1. Scale Attention: 각 스케일 feature map에 대한 가중치 계산 (channel-wise)
*   2. Spatial Attention: 합산된 feature map에 공간적 attention 적용 (pixel-wise)
*****************************************************************************************
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFModule(nn.Module):
    """
    Adaptive Scale Fusion Module for DBNet++

    in_channels: UNet의 output_channels * num_scales (concat된 채널 수)
    num_scales: UNet에서 나오는 feature map의 수 (기본 4)
    """
    def __init__(self, in_channels=256, num_scales=4):
        super(ASFModule, self).__init__()
        self.num_scales = num_scales
        # 각 스케일당 채널 수
        self.scale_channels = in_channels // num_scales  # 64

        # Scale Attention: 각 스케일 feature에 대한 가중치 (1x1 conv → sigmoid)
        # 입력: concat된 모든 스케일 feature (in_channels)
        # 출력: num_scales개의 가중치 맵
        self.scale_attn = nn.Sequential(
            nn.Conv2d(in_channels, num_scales, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention: weighted sum 이후 pixel-wise attention
        # 입력: scale_channels (weighted sum 결과)
        # 출력: scale_channels (attention 적용 결과)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(self.scale_channels, self.scale_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.scale_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.scale_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fuse):
        """
        fuse: concat된 multi-scale feature map (N, in_channels, H, W)
              = (N, scale_channels * num_scales, H, W)

        return: (N, in_channels, H, W) - ASF 적용 결과
        """
        # 1. Scale Attention 계산: (N, num_scales, H, W)
        scale_weights = self.scale_attn(fuse)  # (N, num_scales, H, W)

        # 각 스케일 feature 분리: list of (N, scale_channels, H, W)
        scale_features = torch.chunk(fuse, self.num_scales, dim=1)

        # 2. Scale Attention 적용 후 weighted sum: (N, scale_channels, H, W)
        weighted_sum = sum(
            scale_features[i] * scale_weights[:, i:i+1, :, :]
            for i in range(self.num_scales)
        )

        # 3. Spatial Attention 계산 및 적용
        spatial_weight = self.spatial_attn(weighted_sum)  # (N, 1, H, W)
        enhanced = weighted_sum * spatial_weight          # (N, scale_channels, H, W)

        # 4. 각 스케일에 spatial attention 재적용 후 concat
        # enhanced feature를 각 스케일에 residual로 더해서 최종 출력
        out_features = []
        for i in range(self.num_scales):
            # scale attention + spatial attention을 각 스케일에 반영
            out = scale_features[i] * scale_weights[:, i:i+1, :, :] + enhanced
            out_features.append(out)

        return torch.cat(out_features, dim=1)  # (N, in_channels, H, W)
