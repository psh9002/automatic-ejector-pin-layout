import cadquery as cq
from cadquery import exporters
from cairosvg import svg2png
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import argparse
import sys
import os

Image.MAX_IMAGE_PIXELS = None

def crop_and_resize(img, bbox, img_size):

    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    if (bbox[1]-bbox[0]) > (bbox[3]-bbox[2]):
        img = cv2.rotate(img , cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, img_size)
    return img


dataset_path = sys.argv[1]
product_type = dataset_path.split("/")[-1]

render_img_size = (1920*5, 1080*5)
save_img_size = (1920*3, 1080*3) 


dataset_path = dataset_path

product_paths = reversed(sorted(glob.glob(dataset_path + "/*")))
for product_path in tqdm(product_paths):
    if os.path.exists(product_path + "/png_tmp/outside_all_line.png"):
        print("Already exists at {}! SKipping this ...".format(product_path))
        continue

    cad_name = product_path.split("/")[-1]
    cad_file = product_path + "/tree.stp"
    print("==> Processing", cad_file)
    print("Importing")
    result = cq.importers.importStep(cad_file)
    print("Exporting")
    exporters.export(result, 'test.svg', 
                opt={
                    "width": render_img_size[0],
                    "height": render_img_size[1],
                    "marginLeft": 0,
                    "marginTop": 0,
                    "showAxes": False,
                    "projectionDir": (0, 0, -1),
                    "strokeWidth": 0.25,
                    "strokeColor": (255, 0, 0),
                    "hiddenColor": (0, 0, 255),
                    "showHidden": True,
                },)
    with open('test.svg', 'rb') as f:
        img = f.read()
    img = svg2png(img, background_color="black")
    img = Image.open(BytesIO(img)).convert('RGBA')
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    non_zeros = np.where(img[:, :, 2] == 255)

    bbox = np.min(non_zeros[0]), np.max(non_zeros[0]), np.min(non_zeros[1]), np.max(non_zeros[1])
    img = crop_and_resize(img, bbox, save_img_size)
    cv2.imwrite(product_path + "/png_tmp/inside_all_line.png", img)
    print("Image Saved")
    print("Exporting")
    exporters.export(result, 'test.svg', 
                opt={
                    "width": render_img_size[0],
                    "height": render_img_size[1],
                    "marginLeft": 0,
                    "marginTop": 0,
                    "showAxes": False,
                    "projectionDir": (0, 0, 1),
                    "strokeWidth": 0.25,
                    "strokeColor": (255, 0, 0),
                    "hiddenColor": (0, 0, 255),
                    "showHidden": True,
                },)

    with open('test.svg', 'rb') as f:
        img = f.read()
    img = svg2png(img, background_color="black")
    img = Image.open(BytesIO(img)).convert('RGBA')
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    non_zeros = np.where(img[:, :, 2] == 255)
    bbox = np.min(non_zeros[0]), np.max(non_zeros[0]), np.min(non_zeros[1]), np.max(non_zeros[1])
    img = crop_and_resize(img, bbox, save_img_size)
    cv2.imwrite(product_path + "/png_tmp/outside_all_line.png", img)

# exporters.export(result, 'test.svg', 
#             opt={
#                 "width": render_img_size[0],
#                 "height": render_img_size[1],
#                 "marginLeft": 0,
#                 "marginTop": 0,
#                 "showAxes": False,
#                 "projectionDir": (0, 0, 1),
#                 "strokeWidth": 0.25,
#                 "strokeColor": (255, 0, 0),
#                 "hiddenColor": (0, 0, 255),
#                 "showHidden": True,
#             },)

# with open('test.svg', 'rb') as f:
#     img = f.read()
# img = svg2png(img, background_color="black")
# img = Image.open(BytesIO(img)).convert('RGBA')
# img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
# non_zeros = np.where(img[:, :, 2] == 255)
# bbox = np.min(non_zeros[0]), np.max(non_zeros[0]), np.min(non_zeros[1]), np.max(non_zeros[1])
# img = crop_and_resize(img, bbox, save_img_size)
# cv2.imwrite("test_outside-.png", img)