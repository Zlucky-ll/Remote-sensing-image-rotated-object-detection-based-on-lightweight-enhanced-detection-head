# models/ledh.py
import torch
import torch.nn as nn

class CentralDifferenceConv(nn.Module):
    """Central Difference Convolution (CDConv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CentralDifferenceConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.kernel_size = kernel_size
        
    def forward(self, x):

        conv_out = self.conv(x)

                kernel_size = self.kernel_size
        pad = kernel_size // 2
        

        center = x[:, :, pad:-pad, pad:-pad]
        

        diff = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == pad and j == pad:
                    continue
                shifted = torch.roll(x, shifts=(i-pad, j-pad), dims=(2, 3))
                diff += shifted[:, :, pad:-pad, pad:-pad] - center
        

        cdc_out = diff / (kernel_size * kernel_size - 1)
        
        return conv_out + cdc_out


class SEConv(nn.Module):
    """Shared Enhanced Convolution"""
    
    def __init__(self, in_channels, out_channels):
        super(SEConv, self).__init__()
        

        self.std_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        

        self.cd_conv = CentralDifferenceConv(in_channels, out_channels, 3, padding=1)
        

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
     F_out = F_in * (K_C + K_CDC)
        out = self.std_conv(x) + self.cd_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class LEDH(nn.Module):
    """Lightweight Enhanced Detection Head"""
    
    def __init__(self, in_channels=256, num_classes=15, num_anchors=3):
        super(LEDH, self).__init__()
        

        self.channel_reduce = nn.Conv2d(in_channels, 128, 1)

        self.se_conv = SEConv(128, 128)

        self.cls_head = nn.Conv2d(128, num_classes * num_anchors, 1)
        

        self.reg_head = nn.Conv2d(128, 5 * num_anchors, 1)
        

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):

        x = self.channel_reduce(x)
        

        x = self.se_conv(x)
        

        cls_pred = self.cls_head(x)
        reg_pred = self.reg_head(x)
        
        return cls_pred, reg_pred