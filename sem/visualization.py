
import numpy as np
import math
from scipy.linalg import eigh
import os, glob, sys
import re
import cv2
import imageio
import os.path as path
import transforms3d as tf
import shutil
from collections import defaultdict, namedtuple
from matplotlib import colors as mcolors
from transforms3d.euler import mat2euler

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

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
        self.colours[bbox_det[4] % 32, :]
        thickness = 2
        image_bgr = cv2.rectangle(image_bgr, (left, up), (right, down), color, thickness) 
        return image_bgr

    def plot_bbox(self, bbox_trackers, image_bgr, img_id):
        for trk in bbox_trackers:
            self.plot_track_bbox(trk, image_bgr)

    def plot_all(self, image_bgr, bbox_trackers, kps_trackers, img_id):
        # plot bbox
        image_bgr = self.plot_bbox(bbox_trackers, image_bgr, img_id)
        return image_bgr