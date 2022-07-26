from pathlib import Path
from shutil import rmtree
from docker.types import Mount
from pypcd import pypcd

from typing import Collection

from . import Algorithm

import docker

import numpy as np
import open3d as o3d

__all__ = ["PEAC"]


class PEAC(Algorithm.Algorithm):
    def __init__(
        self,
        container_name: str,
        cfg_path: Path,
        pcd_path: Path,
        size: tuple = (640, 480),
    ):
        self.container_name = container_name
        self.cfg_path = cfg_path
        self.pcd_path = pcd_path
        self.pcd_name = Path(self.pcd_path).stem
        self.size = size
        self._cfg = None
        self._alg_input_dir = Path("input")
        self._alg_output_dir = Path("output")
        self._alg_artifact_name = Path(self.pcd_name + "/" + self.pcd_name + ".npy")
        self._parameter_list = (
            "loop",
            "debug",
            "unitScaleFactor",
            "showWindow",
            "stdTol_merge",
            "stdTol_init",
            "depthSigma",
            "z_near",
            "z_far",
            "angleDegree_near",
            "angleDegree_far",
            "similarityDegreeTh_merge",
            "similarityDegreeTh_refine",
            "depthAlpha",
            "depthChangeTol",
            "initType",
            "minSupport",
            "windowWidth",
            "windowHeight",
            "doRefine",
        )

    def _preprocess_input(self) -> Collection[str]:
        pcd_name = self.__preprocess_point_cloud().name
        cfg_name = self._cfg.write(self._alg_input_dir / self.cfg_path.name).name

        return [pcd_name, cfg_name]

    def __preprocess_point_cloud(self) -> Path:
        pcd_path = self._alg_input_dir / (self.pcd_path.stem + ".pcd")

        pcd = o3d.io.read_point_cloud(str(self.pcd_path))

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pcd.points)

        o3d.io.write_point_cloud(str(pcd_path), cloud)

        pcd = pypcd.PointCloud.from_path(str(pcd_path))

        data = pcd.pc_data.view(np.float32)
        meta = pcd.get_metadata()
        meta["height"] = self.size[0]
        meta["width"] = self.size[1]

        pcd = pypcd.make_xyz_point_cloud(data, meta)
        pcd.save_pcd(pcd_path)

        return Path(self.pcd_path)

    def _evaluate_algorithm(self, input_parameters: Collection[str]) -> Path:
        client = docker.from_env()
        input_mount = Mount(
            target="/app/build/input",
            source=str(self._alg_input_dir.absolute()),
            type="bind",
        )

        output_mount = Mount(
            target="/app/build/output",
            source=str(self._alg_output_dir.absolute()),
            type="bind",
        )

        client.containers.run(
            self.container_name,
            " ".join(input_parameters),
            mounts=[input_mount, output_mount],
        )

        return self._alg_output_dir / self._alg_artifact_name

    def _output_to_labels(self, output_path: Path) -> np.ndarray:
        labels = np.load(output_path)

        return labels

    def _clear_artifacts(self):
        rmtree(self._alg_input_dir)
        rmtree(self._alg_output_dir)
