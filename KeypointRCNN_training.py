import os, json, cv2, numpy as np, matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A # Library for augmentations


# https://github.com/pytorch/vision/tree/main/references/detection
from references_detection import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate