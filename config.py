
import torch

DATASET = {
    'dota': {
        'image_size': 1024,
        'crop_size': 1024,
        'overlap': 200,
        'num_classes': 15,
        'class_names': ['PL', 'BD', 'BR', 'GTF', 'SV', 'LV', 'SH', 'TC', 
                       'BC', 'ST', 'SBF', 'RA', 'HA', 'SP', 'HC']
    },
    'hrsc2016': {
        'image_size': 800,
        'num_classes': 1,
        'class_names': ['ship']
    }
}

TRAIN = {
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


MODEL = {
    'backbone': 'resnet50',
    'neck': 'fpn',
    'num_anchors': 3,
    'num_classes': 15,
    'input_size': 1024
}


LOSS = {
    'reg_loss': 'kld',  # KLD loss
    'cls_loss': 'focal',
    'tau': 1.0  # KLD loss 超参数
}


PRUNE = {
    'method': 'lamp',
    'target_sparsity': 0.3,  # 目标稀疏度
    'prune_ratio': 0.2  # 剪枝比例
}