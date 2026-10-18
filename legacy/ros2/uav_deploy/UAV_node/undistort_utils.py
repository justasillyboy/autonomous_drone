#!/usr/bin/env python3
import cv2
import numpy as np


class SimpleUndistorter:
    """Thin wrapper around cv2.undistort that also exposes intrinsics so
    camera_node can publish an accurate CameraInfo message — the fx/fy values
    are the single source of truth for focal length downstream (perception +
    mission control).  See node_split_deploy.md §2 Node 1.
    """

    def __init__(self, calib_path: str):
        data = np.load(calib_path)
        self.camera_matrix = data['camera_matrix'].astype(np.float64)
        # Tolerate either key name that appears in different capture tools.
        dist_key = 'dist_coeff' if 'dist_coeff' in data.files else 'dist_coeffs'
        self.dist_coeffs = data[dist_key].astype(np.float64)
        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])

    def undistort(self, frame):
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
