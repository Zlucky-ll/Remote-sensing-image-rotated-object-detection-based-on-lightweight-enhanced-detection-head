# models/rotated_detector.py
import torch
import torch.nn as nn
import torchvision.models as models

from models.mfip import MFIP
from models.ledh import LEDH
from losses.kld_loss import KLD_loss

class RotatedObjectDetector(nn.Module):

    def __init__(self, num_classes=15, pretrained=True):
        super(RotatedObjectDetector, self).__init__()
        

        backbone = models.resnet50(pretrained=pretrained)
        self.layer1 = nn.Sequential(*list(backbone.children())[:5])  # C3
        self.layer2 = nn.Sequential(*list(backbone.children())[5:6]) # C4  
        self.layer3 = nn.Sequential(*list(backbone.children())[6:7]) # C5
        

        self.mfip = MFIP(in_channels=[256, 512, 1024], out_channels=256)

        self.ledh = LEDH(in_channels=256, num_classes=num_classes)
        

        self.kld_loss = KLD_loss(tau=1.0)
        
    def forward(self, x, targets=None):

        C3 = self.layer1(x)
        C4 = self.layer2(C3)
        C5 = self.layer3(C4)
        

        features = self.mfip([C3, C4, C5])
        

        all_cls = []
        all_reg = []
        
        for feat in features:
            cls_pred, reg_pred = self.ledh(feat)
            all_cls.append(cls_pred)
            all_reg.append(reg_pred)
            
        if self.training and targets is not None:

            return self.compute_loss(all_cls, all_reg, targets)
        else:

            return all_cls, all_reg
            
    def compute_loss(self, cls_preds, reg_preds, targets):

        cls_loss = 0
        reg_loss = 0
        

        for cls_pred in cls_preds:

            cls_loss += nn.functional.binary_cross_entropy_with_logits(
                cls_pred, 
                torch.zeros_like(cls_pred)  # 占位
            )
            

        for reg_pred in reg_preds:
            reg_loss += self.kld_loss(
                reg_pred,
                torch.zeros_like(reg_pred)  # 占位
            )
            
        return cls_loss + reg_loss