
import matplotlib.pyplot as plt
import imgviz
import glob
import cv2
from tqdm import tqdm
import numpy as np

# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import random


def extract_features(img, model):
    # load the image as a 224x224 array
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    img = cv2.resize(img, (224, 224))
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


def view_cluster(cluster, i):
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    imgs = []
    img_names = []
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)[:1080, :, :]
        img_name = file.split("/")[-1].split(".")[0].split("_")[-1]
        img_names.append(img_name)
        imgs.append(img)
    random.shuffle(imgs)
    n_test = int(len(imgs) * 0.1)
    for j, img in enumerate(imgs):
        img_name = img_names[j]
        if j <= n_test:
            cv2.putText(img, img_name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA) 
        else:
            cv2.putText(img, img_name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA) 

    tiled = imgviz.tile(imgs=imgs, border=(255, 255, 255))

    plt.figure(dpi=500)
    plt.subplot(111)
    plt.title("cluster_{}".format(i))
    plt.imshow(tiled)
    plt.axis("off")
    # img = imgviz.io.pyplot_to_numpy()
    plt.savefig("vis/cluster_{}.png".format(i), dpi=500)
    plt.close()

if __name__ == "__main__":
    
    img_paths = glob.glob("vis/Mold_Cover_Rear_*.png")
    
    model = VGG16()
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

    imgs = []
    img_names = []
    data = {}
    for i, img_path in enumerate(tqdm(img_paths)):
        img_name = img_path.split("/")[-1].split(".")[0].split("_")[-1]
        img = cv2.imread(img_path)[:1080, :, :]
        img = cv2.resize(img, (192*2, 108*2))
        imgs.append(img)
        feat = extract_features(img, model)
        data[img_name] = feat
        img_names.append(img_name)

    # get a list of just the features
    feat = np.array(list(data.values()))
    feat = feat.reshape(-1, 4096)
    # reduce the amount of dimensions in the feature vector
    pca = PCA(n_components=50, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)

    # cluster feature vectors
    kmeans = KMeans(n_clusters=8, n_jobs=-1, random_state=22)
    kmeans.fit(x)

    # holds the cluster id and the images { id: [images] }
    groups = {}
    for file, cluster in zip(img_paths, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    for i, cluster in enumerate(groups.keys()):
        view_cluster(cluster, i)

        