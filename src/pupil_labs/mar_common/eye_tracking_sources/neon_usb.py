import importlib
import time

from . import (
    CameraIntrinsics,
    EyeTrackingData,
    EyeTrackingSource,
)


class NeonUSB(EyeTrackingSource):
    def __init__(self):
        super().__init__()
        import os

        from dotenv import load_dotenv

        from pupil_labs.neon_usb import EyeCamera, SceneCamera

        load_dotenv()

        if "NEON_PIPELINE_MODULE_NAME" not in os.environ:
            raise ValueError("Environment variable NEON_PIPELINE_MODULE_NAME not set.")

        neon_pipeline_module_name = os.environ["NEON_PIPELINE_MODULE_NAME"]
        neon_pipeline_class_name = os.environ["NEON_PIPELINE_CLASS_NAME"]
        neon_pipeline_version = os.environ["NEON_PIPELINE_VERSION"]
        neon_pipeline_module = importlib.import_module(neon_pipeline_module_name)
        neon_pipeline_class = getattr(neon_pipeline_module, neon_pipeline_class_name)

        print("Connecting to cameras...")
        scene_cam = SceneCamera()
        scene_cam.exposure = 120
        eye_cam = EyeCamera()
        print("Done.")

        print("Loading intrinsics...")
        intrinsics = scene_cam.get_intrinsics()
        intrinsics = CameraIntrinsics(
            intrinsics.camera_matrix, intrinsics.distortion_coefficients
        )
        print("Done.")
        print("Setting up pipeline...")
        pipeline = neon_pipeline_class(
            pipeline_version=neon_pipeline_version,
            camera_matrix=intrinsics.camera_matrix,
            dist_coefs=intrinsics.distortion_coefficients,
        )
        print("Done.")
        self._scene_cam = scene_cam
        self._eye_cam = eye_cam
        self._intrinsics = intrinsics
        self._pipeline = pipeline

    @property
    def scene_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    def get_sample(self) -> EyeTrackingData:
        ts = time.time()
        scene_frame = self._scene_cam.get_frame()
        eye_frame = self._eye_cam.get_frame()
        gaze = self._pipeline(eye_frame)
        data = EyeTrackingData(time=ts, gaze=gaze, scene=scene_frame.bgr)
        return data

    def close(self):
        self._scene_cam.close()
        self._eye_cam.close()
