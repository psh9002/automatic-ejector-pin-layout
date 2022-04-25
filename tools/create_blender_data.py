import cv2
import bpy
import bpycv
import random
import numpy as np
import glob
from collections import OrderedDict
import tempfile
from boxx import os, withattr, imread
from tqdm import tqdm
import sys

def get_class_name(part_type):

    class_name = {
        "SOLID": "base",
        "Boss": "boss",
        "BOSS": "boss",
        "Main": "base",
        "Hook": "hook",
        "HOOK": "hook",
        "Rib": "rib",
        "RIB": "rib",
        "U-cut": "ucut",
        "U-cut02": "ucut",
        "EP": "ep",
        "DPS": "dps",
        "Guide-Embo": "guide-embo",
        "Embo": "embo",
        "Screwless-Embo": "screwless-embo",
        "COMPOUND": "base",
        "COMPOUND001": "base"
    }
    return class_name[part_type] 

def set_camera(camera_name):
    for scene in bpy.data.scenes:
        bpy.context.window.scene = scene
    for obj in scene.objects:
        obj.select_set(obj.type == 'CAMERA')
        # set scene camera to active for each viewlayer
        if obj.name == camera_name:
            bpy.context.scene.camera = obj
            bpy.context.scene.render.resolution_x = 1920*6
            bpy.context.scene.render.resolution_y = 1080*6
            print("Using", camera_name)
    


# Set paths
argv = sys.argv
argv = argv[argv.index("--") + 1:] 
dataset_path = argv[0]
for d in tqdm(sorted(glob.glob(dataset_path + "/*"))):
    folder_name = d.split("/")[-1]
    print("==> Processing " + folder_name)
    # create part_path_dict: key=class_name, value=part_path
    inst_id_to_filename = []
    part_path_dict = {
        "boss": [],
        "hook": [],
        "rib": [],
        "ucut": [],
        "ep": [],
        "base": [],
        "dps": [],
        "guide-embo": [],
        "screwless-embo": [],
        "embo": []
    }
    for part_path in sorted(glob.glob(d + "/stl/*.stl")):
        part_name = part_path.split("/")[-1].split(".")[0]
        if len(part_name.split("_")) == 2:
            part_type, part_id = part_name.split("_")
        else:
            part_type =  part_name.split(".")[0]
            part_id = 0
        class_name = get_class_name(part_type)
        part_path_dict[class_name].append(part_path)

    # if not os.path.exists("{}/{}".format(d, "png_tmp")):
    #     os.makedirs("{}/{}".format(d, "png_tmp"))
    # else:
    #     print("Skipping", folder_name)
    #     continue
    for camera_name in ["inside", "outside"]:
        set_camera(camera_name)

        # generate inst images per class_name
        for class_name in part_path_dict.keys():
            # remove all MESH objects
            print("current target: {}, num: {}".format(class_name, len(part_path_dict[class_name])))
            if len(part_path_dict[class_name]) == 0:
                print("Skipping this")
                continue
            [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
            inst_id = 0
            for part_path in part_path_dict[class_name]:
                bpy.ops.import_mesh.stl(filepath=part_path)
                print("Loading", part_path)
                obj = bpy.context.active_object
                inst_id += 1
                obj["inst_id"] = inst_id 
                mat = bpy.data.materials.new("PKHG") 
                mat.diffuse_color = (random.random(), random.random(), random.random(), random.random())
                obj.active_material = mat

            # Render RGB, Depth, Instance Mask
            scene = bpy.data.scenes[0]
            render = scene.render
            befor_render_data_hooks = OrderedDict()

            for hook_name, hook in befor_render_data_hooks.items():
                print(f"Run befor_render_data_hooks[{hook_name}]")
                hook()
            befor_render_data_hooks.clear()

            path = tempfile.NamedTemporaryFile().name
            render_result = {}
            render_result["image"] = bpycv.render_image()
            exr_path = path + ".exr"
            with bpycv.set_inst_material(), bpycv.set_annotation_render(), withattr(
                render, "filepath", exr_path
            ):
                print("Render annotation using:", render.engine)
                bpy.ops.render.render(write_still=True)
            render_result["exr"] = bpycv.exr_image_parser.parser_exr(exr_path)
            os.remove(exr_path)
            result = bpycv.exr_image_parser.ImageWithAnnotation(**render_result)

            # save instance map as 16 bit png
            # the value of each pixel represents the inst_id of the object to which the pixel belongs
            print("==> Saving {}/{}/{}_{}_inst.png".format(d, "png_tmp", camera_name, class_name))
            cv2.imwrite("{}/{}/{}_{}_inst.png".format(d, "png_tmp", camera_name, class_name), np.uint8(result["inst"]))
            
            # convert depth units from meters to millimeters
            depth_in_mm = result["depth"] * 1000
            cv2.imwrite("{}/{}/{}_{}_depth.png".format(d, "png_tmp", camera_name, class_name), np.uint16(depth_in_mm))  # save as 16bit png

        # generate all part image
        inst_id = 0
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
        for class_name in part_path_dict.keys():
            if class_name == "ep":
                continue
            for part_path in part_path_dict[class_name]:
                bpy.ops.import_mesh.stl(filepath=part_path)
                print("Loading", part_path)
                obj = bpy.context.active_object
                inst_id += 1
                obj["inst_id"] = inst_id 
                mat = bpy.data.materials.new("PKHG") 
                mat.diffuse_color = (random.random(), random.random(), random.random(), random.random())
                obj.active_material = mat

        # Render RGB, Depth, Instance Mask
        scene = bpy.data.scenes[0]
        render = scene.render
        befor_render_data_hooks = OrderedDict()

        for hook_name, hook in befor_render_data_hooks.items():
            print(f"Run befor_render_data_hooks[{hook_name}]")
            hook()
        befor_render_data_hooks.clear()

        path = tempfile.NamedTemporaryFile().name
        render_result = {}
        render_result["image"] = bpycv.render_image()
        exr_path = path + ".exr"
        with bpycv.set_inst_material(), bpycv.set_annotation_render(), withattr(
            render, "filepath", exr_path
        ):
            print("Render annotation using:", render.engine)
            bpy.ops.render.render(write_still=True)
        render_result["exr"] = bpycv.exr_image_parser.parser_exr(exr_path)
        os.remove(exr_path)
        result = bpycv.exr_image_parser.ImageWithAnnotation(**render_result)
        
        # convert depth units from meters to millimeters
        depth_in_mm = result["depth"] * 1000
        cv2.imwrite("{}/{}/{}_all_depth.png".format(d, "png_tmp", camera_name), np.uint16(depth_in_mm))  # save as 16bit png
        cv2.imwrite("{}/{}/{}_all_inst.png".format(d, "png_tmp", camera_name), np.uint8(result["inst"])) # all inst img

