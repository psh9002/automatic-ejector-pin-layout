import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp

from loader import SamsungDataset
from utils.visualizer import overlay_mask 
from utils.metric import *
from utils.loss import *
from utils.logger import Logger

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neptune', action="store_true")
    
    args = parser.parse_args()
    
    '''configurations'''
    # train config
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
    
    # logger
    # logger = Logger(cfg, is_neptune=args.neptune)


    log_dir = "train_results"
    
    '''seed'''
    np.random.seed(0)
    random.seed(0)
    cudnn.benchmark = True
    torch.manual_seed(0)
    cudnn.enabled=True
    torch.cuda.manual_seed(0)
    
    '''load dataset'''
    train_dataset = SamsungDataset(cfg, split="train")
    test_dataset = SamsungDataset(cfg, split="test")
    
    train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                                    shuffle=True, num_workers=1)
    val_dataloader =    torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                    num_workers=2)

    '''load model'''
    in_channels = len(cfg["input_direction"]) * len(cfg["input_modality"])
    model = smp.create_model(
            arch = cfg["arch"],
            encoder_name = cfg["encoder_name"],
            in_channels = in_channels,
            classes = 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using {} gpu!".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
    model = model.to(device)

    '''optimizer'''
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    '''Loss'''
    Loss = getattr(sys.modules[__name__], cfg["loss"])(**cfg["loss_kwargs"])

    '''train'''
    best_epoch, best_score = 0, 0
    best_metrics = get_initial_metric(best=True)
    
    optimizer.zero_grad()
    optimizer.step()
    iters = len(train_dataloader)

    for epoch in range(cfg["maximum_epoch"]):
        optimizer.step()

        model = model.train()
        pbar = tqdm(train_dataloader)
        train_metrics = get_initial_metric()
    
        for itr, (input, gt, cts, name) in enumerate(pbar):
            optimizer.zero_grad()
            
            pred = model(input.cuda())
            gt = gt.cuda()
            loss = 0.
            loss += Loss(pred, gt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_norm"])
            optimizer.step()

            # calculate metric
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
            train_metrics = compute_metrics(pred, gt, cts.cuda(), train_metrics, cfg["gt_radius"])
            pbar.set_postfix({'train_loss': loss.item()})
            
        train_metrics = get_average_metrics(train_metrics)    
        for key in train_metrics.keys():
            # logger.logging("train/{}".format(key), train_metrics[key])
            print("train | {:<15}| {:<7.3f}".format(key, train_metrics[key]))
        # print(logger)
        
        # evaluation
        if epoch % cfg["test_interval"] == 0:
            model = model.eval()
            val_metrics = get_initial_metric()
            
            for itr, (input, gt, cts, name) in enumerate(val_dataloader):
                with torch.no_grad():
                    pred = model(input.cuda())
                    pred = torch.sigmoid(pred)
                    pred = pred > 0.5
                    gt = gt.cuda()
                val_metrics = compute_metrics(pred, gt, cts.cuda(), val_metrics, cfg["gt_radius"])
            val_metrics = get_average_metrics(val_metrics)    

            if best_score < val_metrics["iou"]:
                best_score, best_metrics, best_epoch = val_metrics["iou"], val_metrics, epoch
                torch.save(model.state_dict(), os.path.join(log_dir, "best_weights.pkl"))
                    
            print("epoch: {}| loss: {:.4f}, best_epoch: {}, best_score: {:.4f}".format(epoch, loss.item(), best_epoch, best_score))
            for key in val_metrics.keys():
                print("val | {:<15}| {:<7.3f}| best {:<15}| {:<7.3f}".format(key, val_metrics[key], key, best_metrics[key]))
            
            ## visualization
            if epoch % cfg["visualize_interval"] == 0:
                torch.save(model.state_dict(), os.path.join(log_dir, "epoch{}.pkl".format(epoch)))
                (input, gt, cts, name) = next(iter(train_dataloader))
                pred = model(input.cuda())
                pred = torch.sigmoid(pred)
                pred = pred > 0.5
                pred = pred[0].permute(1, 2, 0)
                pred = pred.detach().cpu().numpy()
                gt = np.uint8(gt[0].permute(1, 2, 0)*255)

                input = np.repeat(input[0, 0:1, :, :].numpy() * 255, 3, 0)
                input = np.transpose(input, (1, 2, 0))
                pred = cv2.resize(np.uint8(pred*255), (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
                pred = overlay_mask(input.copy(), pred.copy(), "PRED")
                gt = cv2.resize(gt, (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt = overlay_mask(input.copy(), gt.copy(), "GT")
                
                vis_log = os.path.join(log_dir, "train")
                if not os.path.exists(vis_log):
                    os.makedirs(vis_log)
                cv2.imwrite(os.path.join(vis_log, "epoch{}.png".format(epoch)), np.vstack([input, pred, gt]))
                
                ## visualization
                (input, gt, cts, name) = next(iter(val_dataloader))
                pred = model(input.cuda())
                pred = torch.sigmoid(pred)
                pred = pred > 0.5
                pred = pred[0].permute(1, 2, 0)
                pred = pred.detach().cpu().numpy()
                gt = np.uint8(gt[0].permute(1, 2, 0)*255)
                
                input = np.repeat(input[0, 0:1, :, :].numpy() * 255, 3, 0)
                input = np.transpose(input, (1, 2, 0))
                pred = cv2.resize(np.uint8(pred*255), (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
                pred = overlay_mask(input.copy(), pred, "PRED")
                gt = cv2.resize(gt, (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt = overlay_mask(input.copy(), gt, "GT")
                vis_log = os.path.join(log_dir, "val")
                if not os.path.exists(vis_log):
                    os.makedirs(vis_log)
                cv2.imwrite(os.path.join(vis_log, "epoch{}.png".format(epoch)), np.vstack([input, pred, gt]))
                