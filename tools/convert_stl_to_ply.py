import glob
import open3d as o3d
import numpy as np

dataset_path = "/home/seung/Workspace/project/samsung/Ejector-Pin-Estimation-SAMSUNG/datasets/20210714/stl"
output_path = "/home/seung/Workspace/project/samsung/Ejector-Pin-Estimation-SAMSUNG/datasets/20210714/stl"

# for d in sorted(glob.glob(dataset_path + "/*")):
#     for part in sorted(glob.glob(d + "/*.stl")):
part = "/home/seung/Workspace/project/samsung/Ejector-Pin-Estimation-SAMSUNG/datasets/20210714/stl/K16M00640011/SOLID.stl"
print(part)
mesh = o3d.io.read_triangle_mesh(part)
pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices
pcd.colors = mesh.vertex_colors
pcd.normals = mesh.vertex_normals
pcd = pcd.voxel_down_sample(1)
pcd = pcd.uniform_down_sample(3)
print(np.asarray(pcd.points).shape)

o3d.visualization.draw_geometries([pcd])