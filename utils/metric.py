import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
import cv2
import copy

RADIUS_LIST = range(10, 50, 5)

def get_initial_metric(best=False, distance=False):
    default = []
    if best:
        default = 0
    metrics = {"iou": copy.deepcopy(default), "prec": copy.deepcopy(default), "recall": copy.deepcopy(default)}
    if distance:
        metrics.update({"dist_mean": copy.deepcopy(default)})
        for th in RADIUS_LIST:
            metrics.update({"dist_prec_{}".format(th): copy.deepcopy(default), "dist_recall_{}".format(th): copy.deepcopy(default)})
    return metrics    

def compute_metrics(preds, gts, cts, metrics, radius, distance=False):
    
    preds = preds.detach()
    gts = gts.detach()

    intersection = torch.logical_and(preds, gts)
    union = torch.logical_or(preds, gts)
    precision = torch.sum(intersection) / torch.sum(preds)
    recall = torch.sum(intersection) / torch.sum(gts)
    iou = torch.sum(intersection) / torch.sum(union)

    metrics["prec"].append(precision.item())
    metrics["recall"].append(recall.item())
    metrics["iou"].append(iou.item())

    if distance:
        metrics = compute_distance_metric(preds, cts, metrics, radius)

    return metrics

def get_average_metrics(metrics):
    for key in metrics.keys():
        metrics[key] = np.mean(metrics[key])
    return metrics

#https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3
def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.pow(x - y, p).sum(2) ** (1/p)
    
    return dist

def compute_distance_metric(preds, gts, metrics, radius): 
    batch_size = preds.size(0)
    th_list = RADIUS_LIST
    distance = []
    precision = {th: [] for th in th_list}
    recall = {th: [] for th in th_list}
    for i in range(batch_size):
        #TODO: convert to points
        gt_points = gts[i][0].nonzero().double() # m, 2
            
        pred_points = extract_centroids_cpu(preds[i][0], radius)
        
        if pred_points.size(0) > 0 and gt_points.size(0) > 0:
            dist_mat = distance_matrix(pred_points.cuda(), gt_points)
            dist, near_gts = torch.min(dist_mat, dim=1) # nearest distance of each predict point

            for th in th_list:
                collect = dist < th
                collected_gts = near_gts[collect]
                collected_gts = torch.unique(collected_gts)
                
                precision[th].append(collect.sum().item()/collect.size(0))
                recall[th].append((collected_gts.size(0)/gt_points.size(0)))

            distance.append(torch.mean(dist).item())
                
        else:
            distance.append(0)
    
    metrics["dist_mean"].append(np.mean(distance) if len(distance) > 0 else 0) 
    for th in th_list:
        metrics["dist_prec_{}".format(th)].append(np.mean(precision[th]) if len(precision[th]) > 0 else 0)
        metrics["dist_recall_{}".format(th)].append(np.mean(recall[th]) if len(recall[th]) > 0 else 0)

    return metrics

def extract_nonzero(pred):
    return torch.nonzero(pred)

def extract_centroids_gpu(pred, radius):
    from torchpq.clustering import KMeans

    circle_area = np.pi * radius**2
    pred_points = torch.nonzero(pred)
    k = round(len(pred_points)/circle_area)
    k = np.clip(k, 1, min(500, len(pred_points)))

    pred_points = pred_points.transpose(0, 1).float()
    pred_points = torch.Tensor(pred_points.cpu().numpy()).cuda()


    kmeans = KMeans(n_clusters=k, distance="euclidean")
    labels = kmeans.fit(pred_points)
    centroids = kmeans.compute_centroids(pred_points, labels)
    
    del pred_points, labels, kmeans
    torch.cuda.empty_cache()
    return centroids.transpose(0, 1).double()

def extract_centroids_cupy(pred, radius):
    circle_area = np.pi * radius**2
    pred_points = torch.nonzero(pred)
    k = round(len(pred_points)/circle_area)
    k = np.clip(k, 1, min(500, len(pred_points)))

    pred_points = pred_points.float().cpu().numpy()
    centroids, _ = fit_xp(pred_points, k, 100)
    
    centroids = torch.Tensor(centroids).cuda()

    return centroids.double()

def extract_centroids_cpu(pred, radius, return_contour=False):
    from sklearn.cluster import KMeans
    circle_area = np.pi * radius**2
    
    device = pred.device

    pred = pred.cpu().numpy()
    pred = np.uint8(pred*255)
    contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pred_points = []

    filtered_contours = []
    for cnt in contours:
        A = cv2.contourArea(cnt)
        k = round(A / circle_area)

        if k > 0:
            filtered_contours.append(cnt)
            mask = np.zeros_like(pred)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            pixel_points = mask.nonzero()
            pixel_points = np.array(pixel_points, dtype=np.float32).transpose()

            kmean = KMeans(n_clusters=k)
            kmean.fit(pixel_points)

            for center in kmean.cluster_centers_:
                pred_points.append(center)

    if return_contour:
        return torch.Tensor(pred_points).to(device).double(), filtered_contours

    return torch.Tensor(pred_points).to(device).double()


def fit_xp(X, n_clusters, max_iter):
    import cupy
    assert X.ndim == 2

    # Get NumPy or CuPy module from the supplied array.
    X = cupy.asarray(X)
    xp = cupy.get_array_module(X)
    n_samples = len(X)

    # Make an array to store the labels indicating which cluster each sample is
    # contained.
    pred = xp.zeros(n_samples)

    # Choose the initial centroid for each cluster.
    initial_indexes = xp.random.choice(n_samples, n_clusters, replace=False)
    centers = X[initial_indexes]
    
    for _ in range(max_iter):
        # Compute the new label for each sample.
        distances = xp.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_pred = xp.argmin(distances, axis=1)

        # If the label is not changed for each sample, we suppose the
        # algorithm has converged and exit from the loop.
        if xp.all(new_pred == pred):
            break
        pred = new_pred

        # Compute the new centroid for each cluster.
        i = xp.arange(n_clusters)
        mask = pred == i[:, None]
        sums = xp.where(mask[:, :, None], X, 0).sum(axis=1)
        counts = xp.count_nonzero(mask, axis=1).reshape((n_clusters, 1))
        centers = sums / counts

    return centers, pred