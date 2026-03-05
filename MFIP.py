# models/mfip.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFIP(nn.Module):
    """Multi-layer Feature Interaction Pyramid"""
    
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256):
        super(MFIP, self).__init__()
        

        self.conv1 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels, 1)
        

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.MaxPool2d(2)
        

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        """
        Args:
            features: [C3, C4, C5] from backbone
        Returns:
            fused_features: [P3, P4, P5] enhanced features
        """
        C3, C4, C5 = features
        
        # 通道对齐
        F1 = self.conv1(C3) 
        F2 = self.conv2(C4)  
        F3 = self.conv3(C5)  
                target_size = F2.shape[-2:]
        
        F1_down = F.interpolate(F1, size=target_size, mode='bilinear', align_corners=False)
        F3_up = F.interpolate(F3, size=target_size, mode='bilinear', align_corners=False)
        
 
        Fm = torch.cat([F1_down, F2, F3_up], dim=1)
        Fm = self.fusion_conv(Fm)

        f1 = torch.cat([
            F.interpolate(Fm, size=F1.shape[-2:], mode='bilinear', align_corners=False), 
            F1
        ], dim=1)
        f1 = self.fusion_conv(f1)
        
        f2 = torch.cat([
            F.max_pool2d(Fm, 2),
            F3
        ], dim=1)
        f2 = self.fusion_conv(f2)

        fm = torch.cat([
            F.max_pool2d(f1, 2),
            Fm,
            F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        ], dim=1)
        fm = self.fusion_conv(fm)

        f1_prime = torch.cat([
            F.interpolate(Fm, size=F1.shape[-2:], mode='bilinear', align_corners=False),
            f1,
            F.interpolate(fm, size=F1.shape[-2:], mode='bilinear', align_corners=False)
        ], dim=1)
        f1_prime = self.fusion_conv(f1_prime)
        
        f2_prime = torch.cat([
            F.max_pool2d(Fm, 2),
            f2,
            F.max_pool2d(fm, 2)
        ], dim=1)
        f2_prime = self.fusion_conv(f2_prime)
        
        return [f1_prime, fm, f2_prime]