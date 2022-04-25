import glob

import numpy as np
import cv2
import os
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import argparse


def crop_and_resize(img, bbox, img_size):

    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    if (bbox[1]-bbox[0]) > (bbox[3]-bbox[2]):
        img = cv2.rotate(img , cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, img_size, interpolation = cv2.INTER_NEAREST)
    return img



parser = argparse.ArgumentParser()
parser.add_argument('--product_type', type=str, default="Mold_Cover_Rear")
args = parser.parse_args()

# Set paths
dataset_path = "./datasets/" + args.product_type
original_img_size = (1920*6, 1080*6)
overlap = 0.5
new_W = int(original_img_size[0] / 6)
new_H = int(original_img_size[1] / 6)
W_offset = int(new_W * overlap)
H_offset = int(new_H * overlap)
n_iter = int(original_img_size[0] / W_offset)

error_lists = []
for folder_path in tqdm(sorted(glob.glob(dataset_path + "/*"))):
    product_name = folder_path.split("/")[-1]
    print("product_name:", product_name)

    original_img_paths = glob.glob(folder_path + "/png/*.png")
    if not os.path.exists(folder_path + "/png_crop"):
        os.mkdir(folder_path + "/png_crop")

    for original_img_path in original_img_paths:
        if "ep" in original_img_path or "base" in original_img_path:
            continue
        print(original_img_path)
        original_img = cv2.imread(original_img_path)
        for i in range(n_iter):
            for j in range(n_iter):
                new_img_path = original_img_path.replace(".png", "_{}_{}.png".format(i, j))
                new_img_path = new_img_path.replace("/png", "/png_crop")
                new_img = original_img[H_offset*i:H_offset*i+new_H, W_offset*j:W_offset*j+new_W]
                cv2.imwrite(new_img_path, new_img)



