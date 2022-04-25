import glob
import cv2
import numpy as np
import os
from tqdm import tqdm


dataset_path = "./datasets/Mold_Cover_Rear"

for folder_path in tqdm(sorted(glob.glob(dataset_path + "/*"))):
    product_name = folder_path.split("/")[-1]

    inside_line = cv2.imread(folder_path + "/png/inside_all_line.png")
    outside_line = cv2.imread(folder_path + "/png/outside_all_line.png")
    try:
        inside_blue = np.sum(np.where(inside_line[:, :, 0] == 255, 1, 0))
        outside_blue = np.sum(np.where(outside_line[:, :, 0] == 255, 1, 0))
    except:
        print(product_name)
        continue

    if inside_blue > outside_blue:
        print("Revert {} | inside: {}, outside: {}".format(product_name, inside_blue, outside_blue))
        # swap file names
        for file_path in glob.glob(folder_path + "/png/inside*"):
            new_file_path = file_path.replace("inside", "tmp")
            os.rename(file_path, new_file_path)
        
        for file_path in glob.glob(folder_path + "/png/outside*"):
            new_file_path = file_path.replace("outside", "inside")
            os.rename(file_path, new_file_path)
            
        for file_path in glob.glob(folder_path + "/png/tmp*"):
            new_file_path = file_path.replace("tmp", "outside")
            os.rename(file_path, new_file_path)

