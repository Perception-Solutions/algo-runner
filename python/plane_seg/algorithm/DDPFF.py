from pathlib import Path
from shutil import rmtree

from typing import Collection

from . import Algorithm


import numpy as np
import open3d as o3d

__all__ = ["DDPFF"]


class DDPFF(Algorithm.Algorithm):
    def __init__(self, container_name: str, cfg_path: Path, pcd_path: Path):
        self.container_name = container_name
        self.cfg_path = cfg_path
        self.pcd_path = pcd_path
        self._cfg = None
        self._alg_input_dir = Path("ddpff_input")
        self._alg_output_dir = Path("ddpff_output")
        self._alg_artifact_name = Path("planes.txt")
        self._parameter_list = (
            "floodFill.pointThreshold_min",
            "floodFill.pointThreshold_max",
            "floodFill.planeThreshold_flood",
            "floodFill.planeThreshold_merge",
            "floodFill.planeThreshold_flood_max",
            "floodFill.planeThreshold_merge_max",
            "floodFill.angleThresholdFloodFill",
            "floodFill.angleThresholdFloodFill_max",
            "floodFill.normalSampleDistance_min",
            "floodFill.normalSampleDistance_max",
            "floodFill.c_plane",
            "floodFill.c_plane_merge",
            "floodFill.c_point",
            "floodFill.c_angle",
            "floodFill.c_range",
        )

    def _preprocess_input(self) -> Collection[str]:
        pcd_name = self.__preprocess_point_cloud().name
        cfg_name = self._cfg.write(self._alg_input_dir / "ddpff.ini").name

        return [pcd_name, cfg_name]

    def __preprocess_point_cloud(self) -> Path:
        pcd = o3d.io.read_point_cloud(str(self.pcd_path))
        pcd.paint_uniform_color([0, 0, 0])

        pcd_path = str(self._alg_input_dir / self.pcd_path.stem) + ".ply"
        o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=True)

        return Path(pcd_path)

    def _output_to_labels(self, output_path: Path) -> np.ndarray:
        planes = []
        with open(output_path) as f:
            for line in f:
                planes.append(np.asarray([int(x) for x in line.split()]))

        number_of_points = np.asarray(
            o3d.io.read_point_cloud(str(self.pcd_path)).points
        ).shape[0]

        labels = np.zeros(number_of_points, dtype=int)
        s = set()
        for index, plane_indices in enumerate(planes):
            col = np.random.uniform(0, 1, size=(1, 3))

            while tuple(col[0]) in s:
                col = np.random.uniform(0, 1, size=(1, 3))

            s.add(tuple(col[0]))

            if plane_indices.size > 0:
                labels[plane_indices] = index + 1

        return labels

    def _clear_artifacts(self):
        rmtree(self._alg_input_dir)
        rmtree(self._alg_output_dir)

    @staticmethod
    def __preprocess_pcd(pcd_path: Path, output_dir: Path) -> Path:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pcd.paint_uniform_color([0, 0, 0])
        pcd_path = str(output_dir / pcd_path.stem) + ".ply"
        o3d.io.write_point_cloud(pcd_path, pcd, write_ascii=True)
        return Path(pcd_path)
