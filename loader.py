import os
import cv2
import numpy as np
import torch
import random
import imutils
import json
import pickle

from scipy.stats import multivariate_normal
from scipy.ndimage import center_of_mass

import albumentations as A
import torch.utils.data as data


# pickle
def save_to_pickle(data, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def resize_binary_image(image, factor):
    image = np.uint8(image[:, :, 0] > 0) 
    H, W = image.shape[0]//factor, image.shape[1]//factor
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]) // factor
            cY = int(M["m01"] / M["m00"]) // factor
            centers.append([cY, cX])
            
    image = np.zeros((H, W))
    for y, x in centers:
        image[y][x] = 1

    return image

def preprocess(inst_img, img_size, id, factor):
    image = np.where(inst_img==id, 1, 0)  # image
    if image.size == 0:
        image = np.zeros((img_size[1], img_size[0], 1))
    else:
        image = np.uint8(image)[:, :, 0]
        image = cv2.resize(image, (img_size[0], img_size[1]), interpolation=cv2.INTER_NEAREST)
        image = np.expand_dims(image, -1)
    return image

def preprocess_line(img , img_size):

    img = cv2.resize(img, tuple(img_size), cv2.INTER_AREA)[:, :, 0]
    img = np.where(img != 0, 1, 0)
    img = np.expand_dims(img, -1)
    return img

def preprocess_depth(img, img_size):
    img = cv2.resize(img, tuple(img_size), cv2.INTER_AREA)
    depth_max, depth_min = np.max(img), np.min(img)
    if depth_max - depth_min == 0:
        img = np.zeros_like(img)
    else:
        img = (img - depth_min) / (depth_max - depth_min)
    img = np.expand_dims(img[:, :, 0], -1)
    return img

def extract_center_pts(img, scale=1):

    inst_ids = np.unique(img.copy())
    centers = []
    for inst_id in inst_ids:
        cent_y, cent_x, _ = center_of_mass(np.where(img[:, :, ]==inst_id, 1, 0))
        centers.append([int(cent_x), int(cent_y)])
    return centers

def draw_gaussian_center(img, target_pts, cov=3):
    h, w, _ = img.shape
    x, y = np.mgrid[0:h, 0:w]
    pos = np.dstack((x, y))
    
    cov_1 = np.eye(2) * 100
    rv_1 = multivariate_normal((target_pts[1], target_pts[0]), cov_1)
    gaussian_1 = rv_1.pdf(pos) 

    # cov_2 = np.eye(2) * 1000
    # rv_2 = multivariate_normal((target_pts[0], target_pts[1]), cov_2)
    # gaussian_2 = rv_2.pdf(pos) 
    
    gaussian_img = gaussian_1 
    max_val = np.max(gaussian_img)
    gaussian_img = gaussian_img / max_val 
    gaussian_img[np.isnan(gaussian_img)] = 0
    gaussian_img = np.repeat(np.expand_dims(gaussian_img, -1), 3, -1)
    cv2.imwrite("test__.png", np.uint8(gaussian_img) * 255)
    img = img + gaussian_img
    exit()
    return img


class SamsungDataset(data.Dataset):

    def __init__(self, cfg, split="train"):

        self.dataset_path = cfg["dataset_path"]
        self.input_direction = cfg["input_direction"]
        self.input_modality = cfg["input_modality"]
        self.augmentation = True if split=="train" else False
        self.img_size = cfg["img_size"]
        self.gt_target = cfg["gt_target"]
        self.gt_radius = cfg["gt_radius"]
        
        # transformation
        transformation_list = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GridDistortion()
        ]
        self.clahe = A.Compose([
            A.CLAHE(tile_grid_size=(300, 300))
        ])
        self.augmentor = A.Compose(transformation_list)

        data_dir = os.path.join(self.dataset_path, split)
        self.products = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
        print("Target Data {}: number of products: {}".format(
            data_dir, len(self.products)))
        
        self.data = self.load_data()

    def load_data(self):
        data = {
            "input": [],
            "gt": [],
            "center_mask": [],
            "name": [],
        }
        total = len(self.products)

        processed_root = os.path.join(self.dataset_path, "processed")
        os.makedirs(processed_root, exist_ok=True)


        for idx, product in enumerate(self.products):
            print("processing data... {}/{}".format(idx+1, total), end='\r')

            processed_product = os.path.join(processed_root, "{}..pkl".format(product.split("/")[-1]))
            if os.path.isfile(processed_product):
                temp = load_pickle(processed_product)

                data["input"].append(temp["input"])
                data["gt"].append(temp["gt"])
                
                data["center_mask"].append(temp["center_mask"])
                data["name"].append(temp["name"])
                continue

            input_imgs = []
            if "inside" in self.input_direction:
                if "depth" in self.input_modality:
                    inside_depth = cv2.imread(product + "/png/inside_all_depth.png")
                    inside_depth = preprocess_depth(inside_depth, self.img_size)
                    input_imgs.append(inside_depth)
                if "line" in self.input_modality:
                    inside_line = cv2.imread(product + "/png/inside_all_line.png")
                    inside_line = preprocess_line(inside_line, self.img_size)
                    input_imgs.append(inside_line)
                # if "" # 형상 mask 추가
            ep_inst_img = cv2.imread(product + "/png/inside_ep_inst.png")
            ep_inst_img = cv2.resize(ep_inst_img, tuple(self.img_size), interpolation=cv2.INTER_NEAREST)

            center_pts = extract_center_pts(ep_inst_img)
            gt = np.zeros([self.img_size[1], self.img_size[0], 3])
            for center_pt in center_pts:
                if self.gt_target == "circle":
                    gt = cv2.circle(gt, tuple(center_pt), self.gt_radius, (1, 1, 1), -1)
                elif self.gt_target == "gaussian":
                    gt = draw_gaussian_center(gt, tuple(center_pt), self.gt_radius)
            
            input_imgs = np.concatenate(input_imgs, -1)
            input_imgs = torch.Tensor(input_imgs).permute(2, 0 ,1)
            
            gt = torch.Tensor(np.array(gt[:, :, 0:1])).permute(2, 0, 1).to(dtype=torch.float32)

            center_mask = torch.zeros_like(gt)
            center_pts = np.array(center_pts)
            center_mask[0, center_pts[:, 1], center_pts[:, 0]] = 1

            temp = {
                "input": input_imgs,
                "gt": gt,
                "center_mask": center_mask,
                "name": product.split("/")[-1],
            }
            save_to_pickle(processed_product)

            data["input"].append(input_imgs)
            data["gt"].append(gt)
            
            data["center_mask"].append(center_mask)
            data["name"].append(product.split("/")[-1])
        
        return data

    def __getitem__(self, index):
        input_img = self.data["input"][index].numpy()
        gt = self.data["gt"][index].numpy()
        center_mask = self.data["center_mask"][index].numpy()

        if self.augmentation:
            input_img = np.transpose(input_img, (1, 2, 0))
            transformed = self.clahe(image=np.uint8(input_img[:, :, 0:1]*255))
            input_img[:, :, 0:1] = transformed["image"] / 255
                
            transformed = self.augmentor(image=np.uint8(input_img*255),
                mask=np.stack([gt[0], center_mask[0]], axis=-1)*255)
            input_img = np.transpose(transformed["image"] / 255, (2, 0, 1))
            gt = np.transpose(transformed["mask"][:, :, 0:1]/255, (2, 0, 1))
            center_mask = np.transpose(transformed["mask"][:, :, 1:2]/255, (2, 0, 1))
        input_img = torch.Tensor(input_img.copy())
        gt = torch.Tensor(gt.copy())
        center_mask = torch.Tensor(center_mask.copy())
            
        return input_img, gt, center_mask, self.data["name"][index]
        
    def __len__(self):
        return len(self.products)
