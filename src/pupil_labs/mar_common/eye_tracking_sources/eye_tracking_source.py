from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
import numpy.typing as npt


@dataclass
class EyeTrackingData:
    time: int
    """Timestamp of the data sample in nanoseconds."""

    scene_image_distorted: npt.NDArray[np.uint8]
    """Raw and distorted scene image."""

    gaze_scene_distorted: npt.NDArray[np.float64]
    """Gaze point in distorted scene image coordinates"""

    camera_matrix: npt.NDArray[np.float64]
    """Camera matrix used for undistortion."""

    distortion_coefficients: npt.NDArray[np.float64]
    """Distortion coefficients used for undistortion."""

    @cached_property
    def scene_image_undistorted(self) -> npt.NDArray[np.uint8]:
        """Undistorted scene image."""
        return cv2.undistort(
            self.scene_image_distorted,
            self.camera_matrix,
            self.distortion_coefficients,
            newCameraMatrix=self.camera_matrix,
        )  # type: ignore

    @cached_property
    def gaze_scene_undistorted(self) -> npt.NDArray[np.float64]:
        """Gaze point in undistorted scene image coordinates"""
        return (
            cv2.undistortPoints(
                self.gaze_scene_distorted,
                self.camera_matrix,
                self.distortion_coefficients,
                P=self.camera_matrix,
            )
            .squeeze()
            .astype(np.float64)
        )


@dataclass
class CameraIntrinsics:
    camera_matrix: npt.NDArray[np.float64]
    distortion_coefficients: npt.NDArray[np.float64]


class EyeTrackingSource(ABC):
    @cached_property
    @abstractmethod
    def scene_intrinsics(self) -> CameraIntrinsics:
        pass

    @abstractmethod
    def get_sample(self) -> EyeTrackingData:
        pass

    @abstractmethod
    def close(self):
        pass
