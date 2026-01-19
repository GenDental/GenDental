import numpy as np
import open3d as o3d

def read_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points)
    return xyz 

def write_pointcloud(points, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)

def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    assert points.shape[1] == 3, "Input points should be of shape [n, 3]"
    assert transform.shape == (4, 4), "Transform matrix should be of shape [4, 4]"

    # 添加齐次坐标：从 [n, 3] -> [n, 4]
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack([points, ones])  # [n, 4]

    # 应用变换
    transformed = points_hom @ transform.T  # [n, 4]

    # 去掉齐次坐标，返回 [n, 3]
    return transformed[:, :3]