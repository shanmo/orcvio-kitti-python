
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

# plot quadrics

def plot_ellipsoid_plt(ax, init_by_single_view_flag, v, wTq):
    """
    plot an ellipsoid on the given axis
    :param ax: the axis for plotting
    :param init_by_single_view_flag: this flag determines the color
    :param v: the shape of ellipsoid
    :param wTq: size 4x4, transformation matrix from object to world
    of the ellipsoid
    """

    if init_by_single_view_flag:
        color = 'y'
    else:
        color = 'g'

    Q0_star = get_Q0_star(v)
    Q_star = proj_q2w(Q0_star, wTq)

    A, center = extract_param(invAdj(Q_star))

    plot_ellipsoid(A, center, ax, color)

    ax.view_init(azim=30, elev=40)

def get_Q0_star(v):

    V = np.diag(v)

    Q0_star = np.eye(4)
    Q0_star[0:3, 0:3] = V @ V
    Q0_star[-1, -1] = -1

    return Q0_star

def proj_q2w(Q0_star, wTq):

    # object to world
    Q_star = proj_quad(Q0_star, wTq)

    return Q_star

def extract_param(C):
    """
    extract A, center
    C is a ellipsoid/ellipse in primal form
    return A, quad in primal form, and center
    """

    s = np.shape(C)[0]
    A = C[0:s - 1, 0:s - 1]
    g = C[0:s - 1, s - 1]
    b = C[-1, -1]

    A_inv = np.linalg.inv(A)
    center = -A_inv @ g
    A_scaled = A / (g.T @ A_inv @ g - b)

    return A_scaled, center

def invAdj(A):
    """
    compute inverse adjoint
    B = adj^-1(A) = A^-1
    takes in A, adjoint of B
    output B
    """

    return np.linalg.inv(A)

def plot_ellipsoid(A, center, ax, color):
    """
    Plot the ellipsoid equation in "center form"
    (x-c).T * A * (x-c) = 1
    A is 3x3
    center is 1x3
    """

    U, D, V = np.linalg.svd(A)
    rx, ry, rz = 1. / np.sqrt(D)
    u, v = np.mgrid[0:2 * np.pi:20j, -np.pi / 2:np.pi / 2:10j]

    x = rx * np.cos(u) * np.cos(v)
    y = ry * np.sin(u) * np.cos(v)
    z = rz * np.sin(v)

    E = np.dstack([x, y, z])
    E = np.dot(E, V) + center

    x, y, z = np.rollaxis(E, axis=-1)
    opacity = 0.3
    ax.plot_surface(x, y, z, color=color, cstride=1, rstride=1, alpha=opacity)

def proj_quad(A, P):
    """
    projects A using P
    P A P.T
    inputs: A, P
    object -> world
    world -> cam
    cam -> img
    """

    A_proj = P @ A @ P.T

    return A_proj

# plot wireframe

class CarWireframe():
    """
    plotting car wireframe
    """
    def __init__(self, object_shape, f_plot_all_ax):

        self.starmap_keypionts_3d = object_shape

        self.kp_num = 14
        self.keypoints_2d = np.zeros((self.kp_num, 2))
        self.keypoints_3d = np.zeros((self.kp_num, 3))

        # Edges self.keypoints that are to be connected, for visualizing a car wireframe)
        self.edges = [(1, 3), (0, 2)]
        self.edges += [(3, 7), (2, 6)]
        self.edges += [(7, 13), (6, 12)]
        self.edges += [(13, 11), (12, 10)]
        self.edges += [(11, 9), (10, 8)]
        self.edges += [(9, 5), (8, 4)]
        self.edges += [(5, 1), (4, 0)]

        self.edges += [(i, i + 1) for i in range(0, 13, 2)]

        self.kp_trans_dict = {}
        self.init_kp_dict()

        self.starmap_keypionts_2d = None

        # call the plotting function
        self.plot_wireframe(f_plot_all_ax)

    def add_lines_to_ax_2d(self, ax):

        w2D = self.keypoints_2d.T

        lines2D = [[(w2D[0, i], w2D[1, i]), (w2D[0, j], w2D[1, j])] for (i, j) in self.edges]
        lc2D = LineCollection(lines2D, linewidths=4, colors='r')
        ax.add_collection(lc2D)

    def add_lines_to_ax_3d(self, ax):

        lines3D = [
            [(self.keypoints_3d[i, 0], self.keypoints_3d[i, 1],
              self.keypoints_3d[i, 2]), (self.keypoints_3d[j, 0],
            self.keypoints_3d[j, 1], self.keypoints_3d[j, 2])]
            for (i, j) in self.edges]

        lc3D = Line3DCollection(lines3D)
        ax.add_collection(lc3D)

    def starmap_to_shape_kps_2d(self):

        for part_id in range(np.shape(self.starmap_keypionts_2d)[0]):

            shape_kp_id = self.kp_trans_dict[part_id]
            kp = self.starmap_keypionts_2d[part_id, :]

            self.keypoints_2d[shape_kp_id, 0] = kp[0]
            self.keypoints_2d[shape_kp_id, 1] = kp[1]

    def starmap_to_shape_kps_3d(self):

        for part_id in range(np.shape(self.starmap_keypionts_3d)[0]):

            shape_kp_id = self.kp_trans_dict[part_id]
            kp = self.starmap_keypionts_3d[part_id, :]

            self.keypoints_3d[shape_kp_id, 0] = kp[0]
            self.keypoints_3d[shape_kp_id, 1] = kp[1]
            self.keypoints_3d[shape_kp_id, 2] = kp[2]

    def infer_shape_kp_2d(self):

        """side view mirror"""
        # % 8  ->  'L_SideViewMirror'
        self.keypoints_2d[8, :] = (self.keypoints_2d[10, :] + self.keypoints_2d[4, :]) / 2
        # % 9 ->  'R_SideViewMirror'
        self.keypoints_2d[9, :] = (self.keypoints_2d[11, :] + self.keypoints_2d[5, :]) / 2

    def infer_shape_kp_3d(self):

        """side view mirror"""
        # % 8  ->  'L_SideViewMirror'
        self.keypoints_3d[8, :] = (self.keypoints_3d[10, :] + self.keypoints_3d[4, :]) / 2
        # % 9 ->  'R_SideViewMirror'
        self.keypoints_3d[9, :] = (self.keypoints_3d[11, :] + self.keypoints_3d[5, :]) / 2

    def init_kp_dict(self):

        # % 11 ->  'R_F_RoofTop'
        self.kp_trans_dict[0] = 11
        # % 10 ->  'L_F_RoofTop'
        self.kp_trans_dict[1] = 10
        # % 12 ->  'L_B_RoofTop'
        self.kp_trans_dict[2] = 12
        # % 13 ->  'R_B_RoofTop'
        self.kp_trans_dict[3] = 13
        # % 5  ->  'R_HeadLight'
        self.kp_trans_dict[4] = 5
        # % 4  ->  'L_HeadLight'
        self.kp_trans_dict[5] = 4
        # % 6  ->  'L_TailLight'
        self.kp_trans_dict[6] = 6
        # % 7  ->  'R_TailLight'
        self.kp_trans_dict[7] = 7
        # % 1  ->  'R_F_WheelCenter'
        self.kp_trans_dict[8] = 1
        # % 3  ->  'R_B_WheelCenter'
        self.kp_trans_dict[9] = 3
        # % 0  ->  'L_F_WheelCenter'
        self.kp_trans_dict[10] = 0
        # % 2  ->  'L_B_WheelCenter'
        self.kp_trans_dict[11] = 2

    def plot_wireframe(self, f_plot_all_ax):
        """
        plot the car wireframe on the given axis
        """

        self.starmap_to_shape_kps_3d()
        self.infer_shape_kp_3d()
        self.add_lines_to_ax_3d(f_plot_all_ax)

class FeatureTrackingVis():
    """
    this class visualizes the tracking results
    """

    def __init__(self, img_w, img_h):        
        np.random.seed(0)
        self.colours = np.random.rand(32, 3)  # used only for display
        self.img_w = img_w
        self.img_h = img_h 

    def plot_semantic_kps(self, ax_track, bbox_trackers, kps_trackers):
        cur_updated_obj_id_list = []
        for trk in bbox_trackers:
            # get object id
            bbox_track_id = trk[4]
            cur_updated_obj_id_list.append(bbox_track_id)

        for object_id, trk in kps_trackers.trackers.items():
            # only plot objects visible in cur frame
            if object_id not in cur_updated_obj_id_list:
                continue
            self.plot_sem_tracks(trk, ax_track)

    def plot_sem_tracks(self, kps_tracker, ax_track):
        """
        plot semantic keypoint tracks
        :param kps_tracker:
        :param ax_track:
        :return:
        """
        old_points = defaultdict(dict)
        for idx, track in kps_tracker.history.items():
            for part_id in range(kps_tracker.kp_num):
                # we set kps to nan so need to replace them with 0
                # for plotting
                track_temp = np.copy(track)
                track_temp = np.nan_to_num(track_temp)
                x, y = track_temp[part_id, 0], track_temp[part_id, 1]
                # check whether the feat is inited
                if (x + y) == 0:
                    continue
                x = np.round(x).astype("int")
                y = np.round(y).astype("int")
                # check whether car wireframe is within image border
                if x < 0 or x > self.img_w or y < 0 or y > self.img_h:
                    continue

                ax_track.scatter(x, y, s=1, c=[self.colours[part_id]], marker='s')
                # connect cur kp to next kp
                if part_id in old_points:
                    x_old, y_old = old_points[part_id]
                    lines2D = [[(x_old, y_old), (x, y)]]
                    lc2D = LineCollection(lines2D, colors=mcolors.to_rgba(self.colours[part_id]))
                    ax_track.add_collection(lc2D)

                # update old points
                old_points[part_id] = [x, y]

        for part_id in old_points:
            kp_2d = old_points[part_id]
            # mark the newest kp detection
            ax_track.scatter(kp_2d[0], kp_2d[1], s=50, c=[self.colours[part_id]], marker='v')

    def plot_all(self, image_bgr, curr_tracked_feat_id, obs_all_dict,
                 bbox_trackers, kps_trackers, img_id):
        # plot bbox
        self.plot_bbox(bbox_trackers, image_bgr, img_id)