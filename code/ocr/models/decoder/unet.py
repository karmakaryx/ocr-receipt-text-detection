'''
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/decoders/seg_detector.py
*
* DBNet++ 변경 사항:
*   - ASFModule 추가: multi-scale feature를 concat한 후 ASF 적용
*   - use_asf 플래그로 DBNet / DBNet++ 선택 가능
*
* 참고 논문:
* Real-time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion
* https://arxiv.org/pdf/2202.10304.pdf
*****************************************************************************************
'''

import torch
import torch.nn as nn
from itertools import accumulate
from .asf import ASFModule


class UNet(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 strides=[4, 8, 16, 32],
                 inner_channels=256,
                 output_channels=64,
                 bias=False,
                 use_asf=True):
        super(UNet, self).__init__()

        assert len(strides) == len(in_channels), "Mismatch in 'strides' and 'in_channels' lengths."

        self.use_asf = use_asf
        self.num_scales = len(in_channels)  # 4
        self.output_channels = output_channels

        # UNet 구조 (DBNet과 동일)
        upscale_factors = [strides[idx] // strides[idx - 1] for idx in range(1, len(strides))]
        outscale_factors = list(accumulate(upscale_factors, lambda x, y: x * y))

        self.upsamples = nn.ModuleList()
        for upscale in upscale_factors:
            self.upsamples.append(nn.Upsample(scale_factor=upscale, mode='nearest'))

        self.inners = nn.ModuleList()
        for in_channel in in_channels:
            self.inners.append(nn.Conv2d(in_channel, inner_channels, kernel_size=1, bias=bias))

        self.outers = nn.ModuleList()
        for outscale in reversed(outscale_factors):
            outer = nn.Sequential(nn.Conv2d(inner_channels, output_channels,
                                            kernel_size=3, padding=1, bias=bias),
                                  nn.Upsample(scale_factor=outscale, mode='nearest'))
            self.outers.append(outer)
        self.outers.append(nn.Conv2d(inner_channels, output_channels, kernel_size=3,
                                     padding=1, bias=bias))

        self.upsamples.apply(self.weights_init)
        self.inners.apply(self.weights_init)
        self.outers.apply(self.weights_init)

        # DBNet++: ASF Module
        # concat된 out_features의 총 채널 수 = output_channels * num_scales
        if self.use_asf:
            self.asf = ASFModule(
                in_channels=output_channels * self.num_scales,
                num_scales=self.num_scales
            )

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, features):
        in_features = [inner(feat) for feat, inner in zip(features, self.inners)]

        up_features = []
        up = in_features[-1]
        for i in range(len(in_features) - 1, 0, -1):
            up = self.upsamples[i - 1](up) + in_features[i - 1]
            up_features.append(up)

        out_features = [self.outers[0](in_features[-1])]
        out_features += [outer(feat) for feat, outer in zip(up_features, self.outers[1:])]

        # DBNet: out_features 그대로 반환 (list)
        # DBNet++: concat 후 ASF 적용, DBHead에서 단일 tensor로 처리
        if self.use_asf:
            # 모든 out_features는 같은 H, W (outers에서 upsample 완료)
            fuse = torch.cat(out_features, dim=1)  # (N, output_channels * num_scales, H, W)
            fuse = self.asf(fuse)                  # (N, output_channels * num_scales, H, W)
            return [fuse]  # DBHead의 torch.cat(features, dim=1)에 맞게 list로 감싸기
        else:
            return out_features
