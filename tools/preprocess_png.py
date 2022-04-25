import glob

import numpy as np
import cv2
import os
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import argparse
import sys

def crop_and_resize(img, bbox, img_size):
    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    if (bbox[1]-bbox[0]) > (bbox[3]-bbox[2]):
        img = cv2.rotate(img , cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, img_size, interpolation = cv2.INTER_NEAREST)
    return img



dataset_path = sys.argv[1]
product_type = dataset_path.split("/")[-1]

# Set paths
img_size = (1920*3, 1080*3)

error_lists = []
for folder_path in tqdm(sorted(glob.glob(dataset_path + "/*"))):

    product_name = folder_path.split("/")[-1]
    print("product_name:", product_name)
    print("==>", folder_path + "/png_tmp/outside_all_depth.png")
    depth_img = cv2.imread(folder_path + "/png_tmp/outside_all_depth.png")
    non_zeros = np.where(depth_img != 0)
    bbox = np.min(non_zeros[0]), np.max(non_zeros[0]), np.min(non_zeros[1]), np.max(non_zeros[1])
    outside_depth = crop_and_resize(depth_img, bbox, img_size)

    depth_img = cv2.imread(folder_path + "/png_tmp/inside_all_depth.png")
    non_zeros = np.where(depth_img != 0)
    bbox = np.min(non_zeros[0]), np.max(non_zeros[0]), np.min(non_zeros[1]), np.max(non_zeros[1])
    inside_depth = crop_and_resize(depth_img, bbox, img_size)
    

    if not os.path.exists(folder_path + "/png"):
        os.mkdir(folder_path + "/png")

    if product_type == "Mold_Cover_Rear" or product_type == "Mold_Chassis_Rear" or product_type == "Presentation":
        outside_path = glob.glob(folder_path + "/png_tmp/outside_all*") + glob.glob(folder_path + "/png_tmp/outside_ucut*") + glob.glob(folder_path + "/png_tmp/outside_hook*") + glob.glob(folder_path + "/png_tmp/outside_boss*") + glob.glob(folder_path + "/png_tmp/outside_ep*")
        inside_path = glob.glob(folder_path + "/png_tmp/inside_all*") + glob.glob(folder_path + "/png_tmp/inside_ucut*") + glob.glob(folder_path + "/png_tmp/inside_hook*") + glob.glob(folder_path + "/png_tmp/inside_boss*") + glob.glob(folder_path + "/png_tmp/inside_ep*")
    elif product_type == "Press_Chassis_Rear":
        outside_path = glob.glob(folder_path + "/png_tmp/outside_all*") + glob.glob(folder_path + "/png_tmp/outside_embo*") + glob.glob(folder_path + "/png_tmp/outside_guide*") + glob.glob(folder_path + "/png_tmp/outside_screw*") + glob.glob(folder_path + "/png_tmp/outside_dps*")
        inside_path = glob.glob(folder_path + "/png_tmp/inside_all*") + glob.glob(folder_path + "/png_tmp/inside_embo*") + glob.glob(folder_path + "/png_tmp/inside_guide*") + glob.glob(folder_path + "/png_tmp/inside_screw*") + glob.glob(folder_path + "/png_tmp/inside_dps*")


    for file_path in outside_path:
       
        img = cv2.imread(file_path)
        if "line" not in file_path:
            img = crop_and_resize(img, bbox, img_size)
        else:
            non_zeros = np.where(img[:, :, 0] != 0)
            line_bbox = np.min(non_zeros[0]), np.max(non_zeros[0]), np.min(non_zeros[1]), np.max(non_zeros[1])
            img = crop_and_resize(img, line_bbox, img_size)
            img = np.flip(img, axis=0)
            img = np.flip(img, axis=1)

        cv2.imwrite(folder_path + "/png/" + file_path.split("/")[-1], img)

    for file_path in inside_path:

        img = cv2.imread(file_path)
        if "line" not in file_path:
            img = crop_and_resize(img, bbox, img_size)
        else:
            non_zeros = np.where(img[:, :, 0] != 0)
            line_bbox = np.min(non_zeros[0]), np.max(non_zeros[0]), np.min(non_zeros[1]), np.max(non_zeros[1])
            img = crop_and_resize(img, line_bbox, img_size)
            img = np.flip(img, axis=0)
        cv2.imwrite(folder_path + "/png/" + file_path.split("/")[-1], img)

    # verify 
    imgs = []
    imgs.append(cv2.imread(folder_path + "/png/outside_all_depth.png"))
    imgs.append(cv2.imread(folder_path + "/png/outside_all_line.png"))
    imgs.append(cv2.imread(folder_path + "/png/inside_all_depth.png"))
    imgs.append(cv2.imread(folder_path + "/png/inside_all_line.png"))
    imgs = np.hstack(imgs)
    imgs = cv2.resize(imgs, (1920*4, 1080))
    cv2.imwrite("vis/{}_{}.png".format(product_type, product_name), imgs)


