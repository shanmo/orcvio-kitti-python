import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag
import itertools
from copy import copy, deepcopy
import logging
import os, sys
from filterpy.kalman import KalmanFilter

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../")

import sem.myobject

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", 'INFO'))


class KalmanKpsTracker(object):

    def __init__(self, id):

        # id is same with bbox tracker id
        self.id = id

        self.kp_num = sem.myobject.NUM_KEYPOINTS

        # define constant velocity model
        # x default = [0, 0, 0…0]
        self.kf = KalmanFilter(dim_x = self.kp_num * 4, dim_z = self.kp_num * 4)

        self.dt = 1
        self.kf.F = self.init_F()

        # measurement uncertainty
        self.kf.R = self.init_R()
        # init state uncertainty
        self.kf.P = self.init_P()
        # process noise
        self.kf.Q = self.init_Q()

        self.reset()

        # no need to reset these

        self.all_inited_list = []
        self.history = {}

    def reset(self):

        self.z = np.zeros((self.kp_num * 4, 1))
        self.cur_updated_list = []

        # need to reset H every time
        # since we have different kp obs
        self.kf.H = self.init_H()

    def init_Q(self):
        """
        trust prediction more
        to make kps traj more smooth
        """

        Q_scaler_vel_factor = 1e-2
        Q = np.eye(self.kp_num * 4) * Q_scaler_vel_factor

        # for part_id in range(self.kp_num):
        #     idx = part_id * 4
        #     Q[idx+1, idx+1] *= Q_scaler_vel_factor
        #     Q[idx+3, idx+3] *= Q_scaler_vel_factor

        return Q

    def init_P(self):

        P_diag = 1e-2
        P = np.diag(P_diag * np.ones(self.kp_num * 4))

        # give high uncertainty to the unobservable initial velocities
        P_scaler_vel_factor = 1e0
        for part_id in range(self.kp_num):
            idx = part_id * 4
            P[idx+1, idx+1] *= P_scaler_vel_factor
            P[idx+3, idx+3] *= P_scaler_vel_factor

        return P

    def init_R(self):
        """
        trust measurement less than prediction
        to ensure a smooth kp track
        """

        R_scaler_feat = 5e-2

        # trust velocity less than feat measurement?
        # cov is R_scaler_feat * R_scaler_vel_factor
        R_scaler_vel_factor = 1e0

        R = np.eye(self.kp_num * 4) * R_scaler_feat

        for part_id in range(self.kp_num):
            idx = part_id * 4
            R[idx+1, idx+1] *= R_scaler_vel_factor
            R[idx+3, idx+3] *= R_scaler_vel_factor

        return R

    def init_F(self):
        """
        Process matrix, assuming constant velocity model
        :return:
        """

        F = np.eye(self.kp_num * 4)

        # for both x, y coords
        for i in range(self.kp_num):
            row = i * 4
            offset = row + 1
            F[row, offset] = self.dt

            row = i * 4 + 2
            offset = row + 1
            F[row, offset] = self.dt

        return F

    def init_H(self):
        """
        Measurement matrix, assuming we measure the kps coordinates
        and velocity (vel from bbox)
        :return:
        """

        H = np.eye(self.kp_num * 4)

        # we do not measure velocity
        for part_id in range(self.kp_num):

            row = part_id * 4
            H[row + 1, row + 1] = 0
            H[row + 3, row + 3] = 0

        return H

    def predict(self):

        self.kf.predict()

    def update(self):
        """
        update all kps at the same time
        :return:
        """

        for i in range(self.kp_num):

            # to handle intermittent observation
            if i not in self.cur_updated_list:
                row = i * 4
                self.kf.H[row, row] *= 0
                self.kf.H[row + 2, row + 2] *= 0

            # ignore kps not inited
            if i not in self.all_inited_list:
                row = i * 4
                self.kf.H[row, row] *= 0
                self.kf.H[row + 1, row + 1] *= 0
                self.kf.H[row + 2, row + 2] *= 0
                self.kf.H[row + 3, row + 3] *= 0

        self.kf.update(self.z)

    def convert_x_to_kps(self):
        """
        convert x to kps to be
        stored in history
        :return:
        """

        # all_kps = np.zeros((2, 12))
        # note, nan is used in object lm, not zero!
        all_kps = np.full([self.kp_num, 2], np.nan)

        for part_id in range(self.kp_num):

            row = part_id * 4

            if self.kf.x[row] == 0 and self.kf.x[row + 2] == 0:
                # this kp is not observed, should remain nan
                continue

            all_kps[part_id, 0] = self.kf.x[row]
            all_kps[part_id, 1] = self.kf.x[row + 2]

        return all_kps

    def add_kp_init(self, kp, part_id):

        row = part_id * 4
        self.kf.x[row] = kp[0]
        self.kf.x[row + 2] = kp[1]

        if part_id not in self.all_inited_list:
            self.all_inited_list.append(part_id)

    def add_kp_detection(self, kp, part_id):
        """
        add a detected kp
        :param kp:
        :param part_id:
        :param cnn_noise:
        :return:
        """

        row = part_id * 4

        self.z[row] = kp[0]
        self.z[row + 2] = kp[1]

        self.mark_as_updated_in_this_frame(part_id)

    def mark_as_updated_in_this_frame(self, part_id):

        if part_id not in self.cur_updated_list:
            self.cur_updated_list.append(part_id)

class KpsTracker():

    def __init__(self):

        # dict of kps trackers
        self.trackers = {}

    def update(self, bbox_trackers, sem_kp_object_ids, semantic_kps, kp_labels, img_id):
        """

        :param bbox_trackers:
        :param sem_kp_object_ids:
        :param semantic_kps:
        :param kp_labels:
        :param img_id:
        :return:
        """

        if len(bbox_trackers) == 0 or len(sem_kp_object_ids) == 0:
            # no bbox detection or no keypoints detected in bbox
            return []

        # init trackers for new objects
        unique_object_list = list(set(sem_kp_object_ids))
        for object_id in unique_object_list:
            if object_id not in self.trackers.keys():
                # this is a new object
                kps_tracker = KalmanKpsTracker(object_id)
                self.trackers[object_id] = kps_tracker

        for idx, kp in enumerate(semantic_kps):
            part_id = kp_labels[idx]
            object_id = sem_kp_object_ids[idx]
            kps_tracker = self.trackers[object_id]

            # init kps if unseen
            # add det if seen
            if part_id in kps_tracker.all_inited_list:
                kps_tracker.add_kp_detection(kp.pt, part_id)
            else:
                kps_tracker.add_kp_init(kp.pt, part_id)

        for trk in bbox_trackers:

            # get object id
            bbox_track_id = trk[4]

            # we may have an object in bbox_track_id but not in sem_kp_object_ids
            if bbox_track_id not in self.trackers:
                continue

            kps_tracker = self.trackers[bbox_track_id]

            kps_tracker.predict()
            kps_tracker.update()
            kps_tracker.reset()

            # append kps to history
            self.trackers[bbox_track_id].history[img_id] = \
                self.trackers[bbox_track_id].convert_x_to_kps()
