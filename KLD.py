
import torch
import torch.nn as nn
import math

class KLD_loss(nn.Module):
    """Kullback-Leibler Divergence Loss for rotated object detection"""
    
    def __init__(self, tau=1.0):
        super(KLD_loss, self).__init__()
        self.tau = tau
        
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: predicted boxes [x, y, w, h, theta] (theta in radians)
            target_boxes: target boxes [x, y, w, h, theta]
        Returns:
            loss: KLD regression loss
        """

        xp, yp, wp, hp, thetap = pred_boxes.unbind(dim=1)
        

        xt, yt, wt, ht, thetat = target_boxes.unbind(dim=1)
        

        dx = xp - xt
        dy = yp - yt
        dtheta = thetap - thetat
        

        term1_part1 = 4 * (dx * torch.cos(thetat) + dy * torch.sin(thetat))**2 / (wt**2 + 1e-9)
        term1_part2 = 4 * (dy * torch.cos(thetat) - dx * torch.sin(thetat))**2 / (ht**2 + 1e-9)
        term1 = term1_part1 + term1_part2
        

        term2 = (hp**2/(wt**2 + 1e-9) * torch.sin(dtheta)**2 + 
                 wp**2/(ht**2 + 1e-9) * torch.sin(dtheta)**2 +
                 hp**2/(ht**2 + 1e-9) * torch.cos(dtheta)**2 +
                 wp**2/(wt**2 + 1e-9) * torch.cos(dtheta)**2)
        

        term3 = torch.log((ht**2)/(hp**2 + 1e-9) + 1e-9) + torch.log((wt**2)/(wp**2 + 1e-9) + 1e-9)
        

        kld = 0.5 * (term1 + term2 + term3 - 2)
        kld = torch.clamp(kld, min=0)  
        

        loss = 1 - 1 / (self.tau + torch.exp(-kld))  
        
        return loss.mean()