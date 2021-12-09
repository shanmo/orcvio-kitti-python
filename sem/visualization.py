
import numpy as np
import math
from scipy.linalg import eigh
import cv2

class FeatureTrackingVis():
    """
    this class visualizes the tracking results
    """

    def __init__(self):        
        np.random.seed(0)
        self.colours = np.random.rand(32, 3)  # used only for display

    def plot_track_bbox(self, bbox_det, image_bgr):
        """
        plot bbox detected by yolo
        """
        left = int(bbox_det[0])
        up = int(bbox_det[1])
        right = int(bbox_det[2])
        down = int(bbox_det[3])
        color = self.colours[int(bbox_det[4]) % 32, :] * 255
        thickness = 5
        image_bgr = cv2.rectangle(image_bgr, (left, up), (right, down), color, thickness) 
        image_bgr = cv2.putText(image_bgr, "ID: " + f'{int(bbox_det[4])}', (left, up), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, color, thickness, cv2.LINE_AA)
        return image_bgr

    def plot_bbox(self, bbox_trackers, image_bgr, img_id):
        for trk in bbox_trackers:
            image_bgr = self.plot_track_bbox(trk, image_bgr)
        return image_bgr

    def plot_all(self, image_bgr, bbox_trackers, kps_trackers, img_id):
        # plot bbox
        image_bgr = self.plot_bbox(bbox_trackers, image_bgr, img_id)
        return image_bgr