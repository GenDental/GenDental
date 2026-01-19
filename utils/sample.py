import trimesh
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_vertex_curvature(mesh, k=12):
    """
    用法向量变化近似估计每个顶点的“曲率”，返回每个顶点的权重值。
    """
    normals = mesh.vertex_normals  # shape: (V, 3)
    vertices = mesh.vertices       # shape: (V, 3)
    curvature = np.zeros(len(vertices))

    # 使用 cKDTree 查询近邻
    kdtree = cKDTree(vertices)

    for i, v in enumerate(vertices):
        dists, idx = kdtree.query(v, k=k)
        neighbor_normals = normals[idx]
        center_normal = normals[i]
        diff = neighbor_normals - center_normal
        curvature[i] = np.mean(np.linalg.norm(diff, axis=1))

    return curvature

def importance_sampling(mesh, n_samples=2048, k=12, temperature=0.01):
    """
    根据曲率变化的重要性权重，采样 mesh 顶点。
    """
    curvature = compute_vertex_curvature(mesh, k)
    weights = np.exp(curvature / temperature)
    probs = weights / np.sum(weights)

    indices = np.random.choice(len(mesh.vertices), size=n_samples, replace=False, p=probs)
    sampled_points = mesh.vertices[indices]

    return sampled_points, indices, curvature

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    indexes = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        indexes[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[indexes.astype(np.int32)]
    return point, indexes

def sample_points(mesh, sample_num):
    pts1, index1 = farthest_point_sample(mesh.vertices, int(sample_num/2))
    pts2, index2, _ = importance_sampling(mesh, int(sample_num/2))
    pts = np.concatenate([pts1, pts2], axis=0)
    index = np.concatenate([index1, index2], axis=0)
    return pts, index