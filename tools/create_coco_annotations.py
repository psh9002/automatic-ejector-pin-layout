import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import json
from pycocotools import mask as m
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import datetime
from PIL import Image


def mask_to_polygon(mask):
    # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contour = measure.find_contours(mask, 0.5, positive_orientation='low')
    contour = contour[0]

    polygons = []

    # Flip from (row, col) representation to (x, y)
    # and subtract the padding pixel
    for i in range(len(contour)):
        row, col = contour[i]
        contour[i] = (col - 1, row - 1)

    # Make a polygon and simplify it
    poly = Polygon(contour)
    poly = poly.simplify(1.0, preserve_topology=False)
    polygons.append(poly)
    segmentation = [np.array(poly.exterior.coords).ravel().tolist()]

    return segmentation


def mask_to_rle(mask):
    rle = m.encode(mask)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def get_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) ==0:
        return None, None, None, None
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)


def create_image_info(image_id, outside_line_file_path, outside_depth_file_path, inside_depth_file_path, inside_line_file_path, height, width):
    return {
        "id": image_id,
        "file_name": outside_line_file_path,
        "depth_file_name": outside_depth_file_path,
        "inside_file_name": inside_line_file_path,
        "inside_depth_file_name": inside_depth_file_path,
        "width": width,
        "height": height,
        "date_captured": datetime.datetime.utcnow().isoformat(' '),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }        


def create_samsung_annotation(data_root, target, mode):


    coco_json = {
            "Mold": {
                "info": {
                    "description": "Samsung Dataset for mold part detection and ejector pin location design ",
                    "url": "https://github.com/gist-ailab/samsung-ejector-pin-location-design",
                    "version": "0.1.0",
                    "year": 2021,
                    "contributor": "Seunghyeok Back, Sungho Bak, Raeyoung Kang",
                    "date_created": datetime.datetime.utcnow().isoformat(' ')
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "Attribution-NonCommercial-ShareAlike License",
                        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                    }
                ],
                "categories": [
                    {
                        'id': 1,
                        'name': 'hook',
                        'supercategory': 'shape',
                    },
                    {
                        'id': 2,
                        'name': 'ucut',
                        'supercategory': 'shape',
                    },
                    {
                        'id': 3,
                        'name': 'boss',
                        'supercategory': 'shape',
                    },
                ],
                "images": [],
                "annotations": []
            },
            "Press": {
                "info": {
                    "description": "Samsung Dataset for mold part detection and ejector pin location design ",
                    "url": "https://github.com/gist-ailab/samsung-ejector-pin-location-design",
                    "version": "0.1.0",
                    "year": 2021,
                    "contributor": "Seunghyeok Back, Sungho Bak, Raeyoung Kang",
                    "date_created": datetime.datetime.utcnow().isoformat(' ')
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "Attribution-NonCommercial-ShareAlike License",
                        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                    }
                ],
                "categories": [
                    {
                        'id': 1,
                        'name': 'dps',
                        'supercategory': 'shape',
                    },
                    {
                        'id': 2,
                        'name': 'embo',
                        'supercategory': 'shape',
                    },
                    {
                        'id': 3,
                        'name': 'guide-embo',
                        'supercategory': 'shape',
                    },
                    {
                        'id': 4,
                        'name': 'screwless-embo',
                        'supercategory': 'shape',
                    },
                ],
                "images": [],
                "annotations": []
            }
        }
    class_name_to_id = {
            "Mold": {
                "hook": 1,
                "ucut": 2,
                "boss": 3
            },
            "Press": {
                "dps": 1,
                "embo": 2,
                "guide-embo": 3,
                "screwless-embo": 4
            }
        }
    class_types = {
        "Mold": ["hook", "boss", "ucut"], 
        "Press": ["dps", "embo", "guide-embo", "screwless-embo"]
        }

    with open(os.path.join(data_root, "split_info.json")) as json_file:
        split_info = json.load(json_file)
   

    coco_ann_name = ''.join((target, "_", mode)) + "_coco_annotation.json"
    coco_json_path = os.path.join(data_root, "annotations", coco_ann_name)

    annotations = []
    image_infos = []
    annotation_id = 1
    img_id = 1

    if target == "Mold":
        folder_names = ["Mold_Chassis_Rear", "Mold_Cover_Rear"]
    elif target == "Press":
        folder_names = ["Press_Chassis_Rear"]
    for folder_name in folder_names:
        for folder_path in tqdm(sorted(split_info[folder_name][mode])):

            folder_path = data_root + "/" + folder_path
            product_name = folder_path.split("/")[-1]
            print("product_name:", product_name)
            for class_type in class_types[target]:
                inst_img = cv2.imread(folder_path + f"/png/outside_{class_type}_inst.png")
                if np.size(inst_img) == 1:
                    print("skip", product_name)
                    continue
                height, width, _ = inst_img.shape
                inst_ids = np.unique(inst_img)
                for inst_id in inst_ids:
                    if inst_id == 0:    # inst_id 0은 배경이라 무시함
                        continue
                    mask_img = np.where(inst_img == inst_id, 1, 0)
                    mask_img = np.array(mask_img[:, :, 0], dtype=bool, order='F')
                    bbox = get_bbox(mask_img)
                    annotation = {}
                    annotation["id"] = annotation_id
                    annotation_id += 1
                    annotation["image_id"] = img_id
                    annotation["category_id"] = class_name_to_id[target][class_type]
                    annotation["bbox"] = bbox
                    annotation["height"] = height
                    annotation["width"] = width
                    annotation["iscrowd"] = 0
                    annotation["segmentation"] = mask_to_polygon(mask_img)
                    annotation["area"] = int(np.sum(mask_img))
                    annotations.append(annotation)

            outside_line_file_path = os.path.join(*folder_path.split("/")[-2:]) + "/png/outside_all_line.png"
            outside_depth_file_path = os.path.join(*folder_path.split("/")[-2:]) + "/png/outside_all_depth.png"
            inside_line_file_path = os.path.join(*folder_path.split("/")[-2:]) + "/png/inside_all_line.png"
            inside_depth_file_path = os.path.join(*folder_path.split("/")[-2:]) + "/png/inside_all_depth.png"
            image_infos.append(create_image_info(img_id, outside_line_file_path, outside_depth_file_path, inside_depth_file_path, inside_line_file_path, height, width))
            img_id += 1
           
    coco_json[target]["annotations"] = annotations
    coco_json[target]["images"] = image_infos
    coco_json = coco_json[target]
    with open(coco_json_path, "w") as f:
        print("Saving annotation as COCO format to", coco_json_path)
        json.dump(coco_json, f, indent=4)
    return coco_json_path

if __name__ == "__main__":
    data_root = "./datasets"

    # create_samsung_annotation(data_root, target="Mold", mode="train")   # class: hook & ucut & boss
    # create_samsung_annotation(data_root, target="Mold", mode="test")
    create_samsung_annotation(data_root, target="Press", mode="train")   # class: hook & ucut & boss
    create_samsung_annotation(data_root, target="Press", mode="test")