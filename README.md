# ejector-pin-location-design

Generate the instance segmentation mask - RGB - Depth from STL file using Blender
![sample](./samples/sample_vis.png)

Estimate Ejector Pin Location Design with Sementic Segmenation
![sample](./samples/sample_pred.png)


## Environment Setup

### Blender
```
sudo apt-get install libopenexr-dev python3-pip
wget https://download.blender.org/release/Blender2.90/blender-2.90.0-linux64.tar.xz
tar -xvf blender-2.90.0-linux64.tar.xz
```

Install [bpycv](https://github.com/DIYer22/bpycv)
```
cd blender-2.90.0-linux64/2.90/python/bin
./python3.7m -m ensurepip  # get pip
./python3.7m -m pip install -U pip setuptools wheel 
./python3.7m -m pip install -U opencv-python openexr bpycv tqdm pillow scikit-image 
```

Fix python error
```
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar -xzf Python-3.7.9.tgz
cp Python-3.7.9/Include/* /path/to/blender-2.90.0-linux64/2.90/python/include/python3.7m/
```


### Python

Install miniconda at [link](https://docs.conda.io/en/latest/miniconda.html)


Deep learning
```
conda create -n samsung python=3.7
conda activate samsung
pip install torch torchvision opencv-python tqdm matplotlib scipy imutils pyyaml
```

Freecad & cad-query
```
conda create -n freecad python=3.8
conda activate freecad
conda install -c conda-forge freecad 
pip install tqdm

conda create -n cadquery python=3.8
conda activate cadquery
conda install -c cadquery -c conda-forge cadquery=master
pip install cairosvg opencv-python tqdm
```

## Getting Started


### Pre-process all

```
./pre_process_all.sh
```

### CAD2PNG
1. convert STEP to STL using FreeCAD
```
python tools/rerun.py tools/convert_stp_to_stl.py ~/Dataset/processed_data/datasets/Mold_Cover_Rear
```

2. Render STL to IMG using Blender
Generate dataset
```
conda activate samsung && ./blender-2.90.0-linux64/blender "dataset_generator.blend" --background --python "tools/create_blender_data.py" -- ~/Dataset/processed_data/datasets/Mold_Cover_Rear
```

3. Create line data using cadquery
```
conda activate cadquery && python tools/create_line_dataset.py ~/Dataset/processed_data/datasets/Mold_Cover_Rear
```

4. Crop and resize IMG
```
conda activate samsung && python tools/preprocess_png.py ~/Dataset/processed_data/datasets/Mold_Cover_Rear
```

5. Train a deep neural network
```
python train.py
```

### COCO annotation for part detection

```
python tools/create_coco_annotations.py
```
To visualize the annotation, run `tools/vis_coco_annotations.ipynb`


### Ejector Pin Estimation

1. Prepare Data in `datasets/train` directory
```
datasets/train
├── K15M00020011
├── K15M00040011
├── K15M00430011
└── ...
```
2. Run `train.py`
```
python train.py
```

3. Train results(inference image, model weights) saved in `train_results`
```
train_results
├── best_weights.pkl
├── epoch100.pkl
├── epoch110.pkl
├── epoch120.pkl
├── epoch130.pkl
├── epoch140.pkl
├── epoch150.pkl
├── ...
├── train
│   ├── epoch100.png
│   ├── epoch110.png
│   ├── epoch120.png
│   ├── epoch130.png
│   ├── epoch140.png
│   ├── epoch150.png
│   └── ...
└── val
    ├── epoch100.png
    ├── epoch110.png
    ├── epoch120.png
    ├── epoch130.png
    ├── epoch140.png
    ├── epoch150.png
    └── ...
```

4. Run `inference.py`

```
python inference.py # inference data in datasets/test with best_weights.pkl
python inference.py --split custom # inference data in datasets/custom with best_weights.pkl
python inference.py --epoch 100 # inference data in datasets/test with epoch100.pkl
```

5. Inference results saved in `inference` directory

```
inference
├── K15M00100011
│   ├── pred_pixel.yaml
│   └── visualize.png
├── K16M01820011
│   ├── pred_pixel.yaml
│   └── visualize.png
│   ...
└── K20M00500011
    ├── pred_pixel.yaml
    └── visualize.png

```


- visualize.png
![sample](./samples/sample_pred.png)

- pred_pixel.yaml
```
0:       # index of estimated pin location
- -489.0 # x pixel (related to center of image)
- 267.0  # y pixel (related to center of image)
1:
- 489.0
- 264.0
2:
- 379.0
- 241.0
...

```
