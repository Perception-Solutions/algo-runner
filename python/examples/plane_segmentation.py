from python.plane_seg.algorithm import DDPFF
from python.plane_seg.visualization import color_planes
from pathlib import Path

import sys
import open3d as o3d


def main(argv):
    container_name = argv[0]
    algorithm = DDPFF(
        container_name=container_name,
        cfg_path=Path("data/ddpff.ini"),
        pcd_path=Path("data/0.ply"),
    )

    labels = algorithm.run()
    pcd = o3d.io.read_point_cloud("data/0.ply")
    colored_pcd = color_planes(pcd, labels)

    o3d.visualization.draw_geometries([colored_pcd])


if __name__ == "__main__":
    main(sys.argv[1:])
