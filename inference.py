import os
import sys
import cv2
import argparse
import yaml
import random
import numpy as np
from tqdm import tqdm
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp

from loader import SamsungDataset
from utils.visualizer import overlay_mask, draw_input
from utils.metric import *
from utils.loss import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', 
                        help='path to target data directory. \
                            if args.split==test >> inference ./datasets/test')
    parser.add_argument('--epoch', default=None, type=int)
    args = parser.parse_args()
    
    log_dir = "inference"
    if args.epoch is not None:
        model_path = "train_results/epoch{}.pkl".format(args.epoch)
    else:
        model_path = "train_results/best_weights.pkl"
    
    # configurations
    '''configurations'''
    cfg = {}
    # data config
    cfg.update({
        "dataset_path": "./datasets",

        "img_size": [1024, 576], # width, height
        "input_direction": ["inside"],
        "input_modality": ["depth"],
        "gt_target": "circle",
        "gt_radius": 15,
    })    
    # model
    cfg.update({
        "arch": "MAnet",
        "encoder_name": "densenet121" # timm
    })
    
    '''seed'''
    np.random.seed(0)
    random.seed(0)
    cudnn.benchmark = True
    torch.manual_seed(0)
    cudnn.enabled=True
    torch.cuda.manual_seed(0)
    
    # load dataset
    dataset = SamsungDataset(cfg, split=args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=1)

    # model
    in_channels = len(cfg["input_direction"]) * len(cfg["input_modality"])
    model = smp.create_model(
            arch = cfg["arch"],
            encoder_name = cfg["encoder_name"],
            in_channels = in_channels,
            classes = 1)

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using {} gpu!".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
    model = model.to(device)

    # model state
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.eval()

    metrics = get_initial_metric(distance=True)
    for itr, (input, gt, cts, name) in enumerate(tqdm(dataloader)):
        name = name[0]
        with torch.no_grad():
            pred = model(input.cuda())
            # pred = torch.sigmoid(pred)
            # pred = pred > 0.5
        metrics = compute_metrics(pred, gt.cuda(), cts.cuda(), metrics, cfg["gt_radius"], distance=True)
        # extract centroids
        centroids, pred_contours = extract_centroids_cpu(pred[0][0], cfg["gt_radius"], return_contour=True)
        dist_mat = distance_matrix(centroids.cuda(), gt[0][0].nonzero().double().cuda())
        centroids_dist, near_gts = torch.min(dist_mat, dim=1) # nearest distance of each predict point
        centroids = centroids.cpu().numpy()
    
        # convert to numpy        
        pred = pred[0].permute(1, 2, 0)
        pred = pred.detach().cpu().numpy()
        gt = np.uint8(gt[0].permute(1, 2, 0)*255)
        cts = np.uint8(cts[0].permute(1, 2, 0)*255)
        
        ## visualization
        #1.1 input
        input = np.repeat(input[0, 0:1, :, :].numpy() * 255, 3, 0)
        input = np.transpose(input, (1, 2, 0))

        intersect = np.logical_and(pred, gt)

        #1.2 input with center point
        cts = cv2.resize(cts, (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
        cts = cv2.dilate(cts, np.ones((4, 4)), iterations=2)
        input_cts = overlay_mask(input.copy(), cts, "", [0, 0, 255])
        
        #1.3 ground turth with center point
        gt = cv2.resize(gt, (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt = overlay_mask(input.copy(), gt, "", [204, 204, 255])
        gt_cts = overlay_mask(gt.copy(), cts, "", [0, 0, 255])
        
        #2.1 ground turth
        gt = gt
        #2.2 draw prediction
        pred = cv2.resize(np.uint8(pred*255), (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_mask = pred.copy()
        pred = overlay_mask(input.copy(), pred_mask, "", [255, 101, 51])
        
        #2.3 draw union and intersection
        union = 0.5 * gt + 0.5 * pred
        intersect = cv2.resize(np.uint8(intersect*255), (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
        iou = overlay_mask(union, intersect, "", [0, 255, 0])
        
        # #3.1 draw center points with contours
        # pred_cts_contour = overlay_mask(input.copy(), pred_mask/2, "", [255, 101, 51])
        # cv2.drawContours(pred_cts_contour, pred_contours, -1, [255, 0, 0], 2)
        # for point in centroids:
        #     point = list(map(round, point))
        #     cv2.circle(pred_cts_contour, (point[1], point[0]), 3, (0, 255, 255), -1)
        #
        # for th in RADIUS_LIST:
        #     cv2.circle(pred_cts_contour, (point[1], point[0]), th, (0, 255, 255), 1)
        #
        # #3.2 draw filtered center points with edges
        # # filter edge
        # canny = cv2.Canny(np.uint8(input[:, :, 0]), 60, 200)
        # dilated_canny = cv2.dilate(canny, np.ones((2,2)), iterations=1)
        # filter_edge = overlay_mask(input.copy(), dilated_canny/2, "", [255, 0, 0])
        # for point in centroids:
        #     point = list(map(round, point))
        #     cv2.circle(filter_edge, (point[1], point[0]), 3, (0, 255, 255), -1)
        #
        # # filtering
        # edge_points = torch.Tensor(np.transpose(dilated_canny.nonzero())).cuda()
        # dist_mat = distance_matrix(torch.Tensor(centroids).cuda(), edge_points)
        # dist, near_gts = torch.min(dist_mat, dim=1)
        # filtered_centroids = []
        # removed_centroids = []
        # filtered_dist = []
        # for idx, point in enumerate(centroids):
        #     point = list(map(round, point))
        #     if dist[idx] < 1: # remove
        #         removed_centroids.append(point)
        #         cv2.circle(filter_edge, (point[1], point[0]), 10, (0, 255, 0), 3)
        #     else:
        #         filtered_centroids.append(point)
        #         filtered_dist.append(centroids_dist[idx].item())
        # #3.3 filtered centers
        # filtered_cts = input.copy()
        # filtered_cts = overlay_mask(input.copy(), dilated_canny/2, "", [255, 0, 0])
        # for point in filtered_centroids:
        #     cv2.circle(filtered_cts, (point[1], point[0]), 3, (0, 255, 255), -1)
        # for point in removed_centroids:
        #     single_point_img = np.zeros_like(input.copy())
        #     cv2.circle(single_point_img, (point[1], point[0]), 10, (255, 255, 255), -1)
        #     single_point_img = cv2.bitwise_and(np.uint8(single_point_img[:, :, 0]), cv2.bitwise_not(np.uint8(dilated_canny)))
        #     contours, hierarchy = cv2.findContours(single_point_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #
        #     depths = []
        #     for contour in contours:
        #         tmp_img = np.zeros_like(input.copy())
        #         cv2.drawContours(tmp_img, [contour], -1, 1, -1)
        #         tmp_img = tmp_img * input.copy()
        #         depths.append(np.sum(tmp_img))
        #     c = contours[np.argmin(depths)]
        #     x,y,w,h = cv2.boundingRect(c)
        #     cv2.circle(filtered_cts, (int(x+w/2), int(y+h/2)), 3, (0, 255, 255), -1)
        #     cv2.circle(filtered_cts, (int(x+w/2), int(y+h/2)), 10, (0, 255, 0), 3)
        #     filtered_centroids.append([int(y+h/2), int(x+w/2)])
        #
        # #4.1 draw distance metric
        # dist_cts = input.copy()
        # filtered_dist = np.array(filtered_dist)
        # # colors = plt.get_cmap("seismic")(filtered_dist/np.max(filtered_dist))
        # colors = plt.get_cmap("seismic")(filtered_dist/50)
        # for point, color in zip(filtered_centroids, colors):
        #     color = np.array(color*255)
        #     cv2.circle(dist_cts, (point[1], point[0]), 10, color, -1)
        #     cv2.circle(dist_cts, (point[1], point[0]), 3, (0, 255, 255), -1)
        # for th in RADIUS_LIST:
        #     cv2.circle(dist_cts, (point[1], point[0]), th, (0, 255, 255), 1)
        #
        # dist_cts = overlay_mask(dist_cts.copy(), cts, "", [0, 0, 255])
        #
        # #4.2 draw radius th
        # rad_cts = input.copy()
        # filtered_dist = np.array(filtered_dist)
        #
        # for point, dist in zip(filtered_centroids, filtered_dist):
        #     for idx, th in enumerate(RADIUS_LIST):
        #         if dist < th:
        #             break
        #     color = np.array(plt.get_cmap("tab10")(idx))[:3]*255
        #     cv2.circle(rad_cts, (point[1], point[0]), 10, color[::-1], -1)
        #     cv2.circle(rad_cts, (point[1], point[0]), 3, (0, 255, 255), -1)
        #
        # for th in RADIUS_LIST:
        #     cv2.circle(rad_cts, (point[1], point[0]), th, (0, 255, 255), 1)
        # rad_cts = overlay_mask(rad_cts.copy(), cts, "", [0, 0, 255])
        #
        # #4.3
        # gt_pred = input.copy()
        #
        # for point in filtered_centroids:
        #     cv2.circle(gt_pred, (point[1], point[0]), 10, [0, 255, 255], -1)
        #
        # for th in RADIUS_LIST:
        #     cv2.circle(gt_pred, (point[1], point[0]), th, (255, 0, 0), 1)
        # gt_pred = overlay_mask(gt_pred.copy(), cts, "", [0, 0, 255])
        # pred_result = {}
        # for i, point in enumerate(filtered_centroids):
        #     pred_result[i] = [point[1]-cfg["img_size"][0]/2, point[0]-cfg["img_size"][1]/2]





        raw1 = np.hstack([input, gt_cts])
        raw2 = np.hstack([pred, iou])
        # raw3 = np.hstack([filtered_cts, gt_pred])

        if not os.path.exists(os.path.join(log_dir, "{}".format(name))):
            os.makedirs(os.path.join(log_dir, "{}".format(name)))

        with open(os.path.join(log_dir, '{}/pred_pixel.yaml'.format(name)), 'w') as f:
            yaml.safe_dump(pred_result, f)

        cv2.imwrite(os.path.join(log_dir, "{}/visualize.png".format(name)), np.vstack([raw1, raw2, raw3]))

    metrics = get_average_metrics(metrics)
    
    for key in metrics.keys():
        try:
            print("------------------------ RADIUS {} ------------------------".format(int(key.split("_")[-1])))
        except:
            pass
        print("Test {:<15}| {:<.3f}".format(key, metrics[key]))