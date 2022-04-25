import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp

import numpy as np
import matplotlib.pyplot as plt
import random


cfg = {
    "batch_size": 4,
    "maximum_epoch": 1000,
    "lr": 0.0001,
    "wd": 0.000001,
    "max_norm": 0.1,

    #TODO: Custom Loss?
    "loss": "DiceLoss", # https://github1s.com/Project-MONAI/MONAI/blob/21a3c16fcba5f99a797ddc6e6595cdf2f18cdd57/monai/losses/dice.py 
    "loss_kwargs": {"sigmoid": True},
    "test_interval": 1,
    "visualize_interval": 10,
}
#TODO: data config
cfg.update({
    "dataset_path": "./datasets",
    "img_size": [1024, 576], # width, height

    "input_direction": ["inside"], 
    "input_modality": ["depth"], # depth or line
    
    "gt_target": "circle",
    "gt_radius": 15,
})    

# model
cfg.update({
    "arch": "MAnet",
    "encoder_name": "densenet121" # timm
})

in_channels = len(cfg["input_direction"]) * len(cfg["input_modality"])

# motivated from keypoint mask, I can get predicted points
# https://learnopencv.com/human-pose-estimation-using-keypoint-rcnn-in-pytorch/


class EP_net(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()

        self.MAnet = smp.create_model(
                                        arch = cfg["arch"],
                                        encoder_name = cfg["encoder_name"],
                                        in_channels = in_channels,
                                        classes = num_classes)

        # self.Clustering_net = 