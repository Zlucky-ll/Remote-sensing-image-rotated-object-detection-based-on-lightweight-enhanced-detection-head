# pruning/lamp.py
import torch
import torch.nn as nn
import numpy as np

class LAMPPruning:
    """Layer-Adaptive Magnitude-based Pruning"""
    
    def __init__(self, model, target_sparsity=0.3):
        self.model = model
        self.target_sparsity = target_sparsity
        
    def compute_lamp_scores(self, layer_weight):
 

        w_flat = layer_weight.view(-1)
        w_sorted, indices = torch.sort(w_flat.abs(), descending=True)

        cumsum = torch.cumsum(w_sorted**2, dim=0)
        

        lamp_scores = w_sorted**2 / cumsum
        

        lamp_scores_orig = torch.zeros_like(w_flat)
        lamp_scores_orig[indices] = lamp_scores
        
        return lamp_scores_orig.view(layer_weight.shape)
    
    def prune_layer(self, layer, keep_ratio):

        if not hasattr(layer, 'weight'):
            return layer
            
        weight = layer.weight.data
        lamp_scores = self.compute_lamp_scores(weight)
        

        flat_scores = lamp_scores.view(-1)
        k = int(flat_scores.numel() * keep_ratio)
        threshold = torch.topk(flat_scores, k, largest=True)[0][-1]
        

        mask = (lamp_scores >= threshold).float()

        layer.weight.data = weight * mask
        
        return layer
    
    def prune(self):

        total_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                total_params += module.weight.numel()
        
                target_params = total_params * (1 - self.target_sparsity)
        
           if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_params = module.weight.numel()
                layer_keep_ratio = target_params / total_params
                self.prune_layer(module, layer_keep_ratio)
                
          pruned = layer_params - (module.weight != 0).sum().item()
                pruned_params += pruned
                
        print(f"LAMP pruning completed. Pruned {pruned_params/total_params*100:.2f}% parameters")
        
        return self.model