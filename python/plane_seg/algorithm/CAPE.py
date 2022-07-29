from pathlib import Path
from shutil import rmtree, copy2

from typing import Collection, Tuple

from . import Algorithm


import numpy as np
import open3d as o3d

__all__ = ["CAPE"]


class CAPE(Algorithm.Algorithm):
    def __init__(
        self, container_name: str, cfg_path: Path, pcd_path: Path, calib_path: Path
    ):
        self.container_name = container_name
        self.cfg_path = cfg_path
        self.pcd_path = pcd_path
        self.calib_path = calib_path
        self._cfg = None
        self._alg_input_dir = Path("input")
        self._alg_output_dir = Path("output")
        self._alg_input_depth_name = Path("depth_0.png")
        self._alg_input_config_name = Path("params.ini")
        self._alg_artifact_name = Path("labels_0.csv")
        self.input_is_depth = pcd_path.suffix == ".png"
        self._parameter_list = (
            "depthSigmaCoeff",
            "depthSigmaMargin",
            "cylinderScoreMin",
            "cylinderRansacSqrMaxDist",
            "cosAngleMax",
            "maxMergeDist",
            "patchSize",
            "minNrOfValidPointsOnePerXThreshold",
            "planesegMaxDiff",
            "planarFittingJumpsCounterThresholdParam",
            "histogramBinsPerCoordParam",
            "regionGrowingCandidateSizeThresholdParam",
            "regionGrowingCellsActivatedThresholdParam",
            "regionPlanarFittingPlanarityScoreThresholdParam",
            "cylinderDetectionCellsActivatedThreshold",
            "refinementMultiplierParam",
        )

    # getting parameters for running algo
    def _preprocess_input(self) -> Collection[str]:
        img_path = self._alg_input_dir / self._alg_input_depth_name
        if self.input_is_depth:
            copy2(self.pcd_path, img_path)
        else:
            depth_image = self.__convert_point_cloud_to_depth_image(shape=(480, 640))
            o3d.io.write_image(str(img_path), depth_image)
        container_cfg_name = str(
            self._cfg.write(self._alg_input_dir / self._alg_input_config_name)
        )

        copy2(self.calib_path, self._alg_input_dir)

        return [self._alg_input_dir, container_cfg_name]

    def __convert_point_cloud_to_depth_image(
        self, shape: Tuple[int, int]
    ) -> o3d.geometry.Image:
        pcd = o3d.io.read_point_cloud(str(self.pcd_path))
        pcd.paint_uniform_color([0, 0, 0])

        xyz_load = np.asarray(pcd.points)
        z = xyz_load[:, 2].reshape(shape[0], shape[1])
        scale_factor = 1000
        depth = (z * scale_factor).astype(np.uint32)
        d_img = o3d.geometry.Image(depth.astype(np.uint16))

        return d_img

    def _output_to_labels(self, output_path: Path) -> np.ndarray:
        labels_table = np.genfromtxt(output_path, delimiter=",").astype(np.uint8)
        labels = labels_table.reshape(labels_table.size)
        return labels

    def _clear_artifacts(self):
        rmtree(self._alg_input_dir)
        rmtree(self._alg_output_dir)
