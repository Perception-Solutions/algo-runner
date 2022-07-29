import open3d as o3d
import numpy as np

__all__ = ["color_planes"]


def color_planes(
    pcd: o3d.geometry.PointCloud, labels: np.ndarray
) -> o3d.geometry.PointCloud:

    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = pcd.points
    colored_pcd.paint_uniform_color([0.0, 0.0, 0.0])

    colors = np.array(colored_pcd.colors)

    label_to_color = {0: np.zeros(shape=(1, 3))}
    color_set = {(0.0, 0.0, 0.0)}

    for label in np.unique(labels):
        if label not in label_to_color:
            col = np.random.uniform(0, 1, size=(1, 3))
            while tuple(col[0]) in color_set:
                col = np.random.uniform(0, 1, size=(1, 3))
            label_to_color[label] = col
            color_set.add(tuple(col[0]))
        colors[labels == label] = label_to_color[label]

    colored_pcd.colors = o3d.utility.Vector3dVector(colors)

    return colored_pcd
