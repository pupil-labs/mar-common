import importlib
import time
from functools import cached_property

import numpy as np

from . import (
    CameraIntrinsics,
    EyeTrackingData,
    EyeTrackingSource,
)


class NeonUSB(EyeTrackingSource):
    def __init__(self, compute_gaze: bool = True):
        super().__init__()
        from pupil_labs.neon_usb import EyeCamera, SceneCamera

        print("Connecting to scene cam...", end="", flush=True)
        self._scene_cam = SceneCamera()
        self._scene_cam.exposure = 120
        print("Done.")

        if compute_gaze:
            print("Connecting to eye cam...", end="", flush=True)
            self._eye_cam = EyeCamera()
            print("Done.")

            import os

            from dotenv import load_dotenv

            load_dotenv()

            if "NEON_PIPELINE_MODULE_NAME" not in os.environ:
                raise ValueError(
                    "Environment variable NEON_PIPELINE_MODULE_NAME not set."
                )

            neon_pipeline_module_name = os.environ["NEON_PIPELINE_MODULE_NAME"]
            neon_pipeline_class_name = os.environ["NEON_PIPELINE_CLASS_NAME"]
            neon_pipeline_version = os.environ["NEON_PIPELINE_VERSION"]
            neon_pipeline_module = importlib.import_module(neon_pipeline_module_name)
            neon_pipeline_class = getattr(
                neon_pipeline_module, neon_pipeline_class_name
            )

            print("Setting up pipeline...", end="", flush=True)
            self._pipeline = neon_pipeline_class(
                pipeline_version=neon_pipeline_version,
                camera_matrix=self.scene_intrinsics.camera_matrix,
                dist_coefs=self.scene_intrinsics.distortion_coefficients,
            )
            print("Done.")
        else:
            self._eye_cam = None
            self._pipeline = None

    @cached_property
    def scene_intrinsics(self) -> CameraIntrinsics:
        intrinsics = self._scene_cam.get_intrinsics()
        intrinsics = CameraIntrinsics(
            intrinsics.camera_matrix, intrinsics.distortion_coefficients
        )
        return intrinsics

    def get_sample(self) -> EyeTrackingData:
        ts = int(time.time() * 1e9)
        scene_frame = self._scene_cam.get_frame()
        if self._pipeline is None:
            gaze = np.array([0, 0], dtype=np.float64)
        else:
            eye_frame = self._eye_cam.get_frame()
            gaze = self._pipeline(eye_frame)
        data = EyeTrackingData(
            time=ts,
            gaze_scene_distorted=gaze,
            scene_image_distorted=scene_frame.bgr,
            camera_matrix=self.scene_intrinsics.camera_matrix,
            distortion_coefficients=self.scene_intrinsics.distortion_coefficients,
        )
        return data

    def close(self):
        self._scene_cam.close()
        if self._eye_cam is not None:
            self._eye_cam.close()
