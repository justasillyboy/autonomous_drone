#!/usr/bin/env python3
import cv2
import numpy as np

class SimpleUndistorter:
    def __init__(self, calib_path):
        data = np.load(calib_path)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeff']
    def undistort(self, frame):
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
