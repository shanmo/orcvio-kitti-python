"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
from numpy import dot
from scipy.linalg import inv
from filterpy.kalman import KalmanFilter

# from sklearn.utils.linear_assignment_ import linear_assignment
# use this in the future
from scipy.optimize import linear_sum_assignment as linear_assignment

# suppress warnings 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def iou(bb_test,bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """

    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)

    return(o)

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x1,y1,x2,y2]
    """

    return bbox.reshape((-1, 1))

def convert_x_to_bbox(x):
    """
    Takes a form [x1,x1 dot, y1, y1 dot, x2, x2 dot, y2, y2 dot] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """

    return np.array([x[0, 0], x[2, 0], x[4, 0], x[6, 0]]).reshape((1, -1))

class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """

    # tracker id starts at 1
    count = 1

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """

        # define constant velocity model
        self.kf = KalmanFilter(dim_x = 8, dim_z = 4)

        self.dt = 1

        self.kf.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                            [0, 1,  0,  0,  0,  0,  0, 0],
                            [0, 0,  1,  self.dt, 0,  0,  0, 0],
                            [0, 0,  0,  1,  0,  0,  0, 0],
                            [0, 0,  0,  0,  1,  self.dt, 0, 0],
                            [0, 0,  0,  0,  0,  1,  0, 0],
                            [0, 0,  0,  0,  0,  0,  1, self.dt],
                            [0, 0,  0,  0,  0,  0,  0,  1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0]])

        # measurement uncertainty/noise
        self.kf.R *= 1e-2

        # give high uncertainty to the unobservable initial velocities
        indices = [1, 3, 5, 7]
        self.kf.P[indices, indices] *= 1e-2

        # process uncertainty
        self.kf.Q *= 1e-2

        indices = [0, 2, 4, 6]
        self.kf.x[indices, 0] = np.squeeze(convert_bbox_to_z(bbox))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count

        # increment tracker id
        KalmanBoxTracker.count += 1

        # keep all states, only publish
        # when object is lost
        # key is image id, value is tracker
        self.history = {}

        self.hit_streak = 0

        # not used
        self.hits = 0
        self.age = 0

    def update_only(self, z):
        '''
        Implement the Kalman Filter, including the predict and the update stages,
        with the measurement z
        '''

        x = self.kf.x

        # Update
        S = dot(self.kf.H, self.kf.P).dot(self.kf.H.T) + self.kf.R
        # Kalman gain
        K = dot(self.kf.P, self.kf.H.T).dot(inv(S))
        # residual
        y = z - dot(self.kf.H, x)

        # x += dot(K, y)
        indices = [0, 2, 4, 6]
        x[indices, 0] = np.squeeze(z)

        self.kf.P = self.kf.P - dot(K, self.kf.H).dot(self.kf.P)
        self.kf.x = x

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # limit bbox velocity
        # self.clip_bbox_vel()

        self.kf.update(convert_bbox_to_z(bbox))
        # self.update_only(convert_bbox_to_z(bbox))

    def clip_bbox_vel(self):

        bbox_vel_threshold = 0

        # print("velocity: u {} v {}".format(self.kf.x[4], self.kf.x[5]))

        indices = [1, 3, 5, 7]
        for id in indices:
            self.kf.x[id, 0] = np.sign(self.kf.x[id, 0]) * min(abs(self.kf.x[id, 0]), bbox_vel_threshold)

    def predict_only(self):
        '''
        Implment only the predict stage. This is used for unmatched detections and
        unmatched tracks
        '''

        x = self.kf.x

        # Predict
        # x = dot(self.kf.F, x)

        self.kf.P = dot(self.kf.F, self.kf.P).dot(self.kf.F.T) + self.kf.Q
        self.kf.x = x

        return convert_x_to_bbox(self.kf.x)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        # limit bbox velocity
        # self.clip_bbox_vel()

        self.kf.predict()
        # self.predict_only()

        self.age += 1

        if (self.time_since_update > 0):
            self.hit_streak = 0

        self.time_since_update += 1

        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """

        return convert_x_to_bbox(self.kf.x)

    def get_vel(self):
        """
        returns velo
        4-5 are u dot v dot based on sort paper
        :return:
        """

        bbox_vel1 = np.array([self.kf.x[1, 0], self.kf.x[3, 0]])
        bbox_vel2 = np.array([self.kf.x[5, 0], self.kf.x[7, 0]])

        bbox_vel = (bbox_vel1 + bbox_vel2) / 2

        # if np.sum(abs(bbox_vel)) == 0:
        #   print("zero velocity!")

        return bbox_vel

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)
    # ref https://github.com/abewley/sort/issues/80
    matched_indices = np.array(list(zip(*matched_indices)))

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):

        # matched_indices may be []
        if np.shape(matched_indices)[0] == 0:
            continue

        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort(object):

    def __init__(self, max_age = 1, min_hits = 3, last_img_id = 0):
        """
        Sets key parameters for SORT
        """

        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

        # to indicate which track is lost
        self.last_img_id = last_img_id
        self.lost_trackers_dict = {}

    def update(self, dets, img_id):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        self.frame_count += 1

        # get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):

                d = matched[np.where(matched[:,1]==t)[0],0]

                # dets may be []
                if np.shape(dets)[0] == 0:
                    continue

                trk.update(dets[d,:][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):

            d = trk.get_state()[0]
            vel = trk.get_vel()

            # two conditions for good track:
            # 1. no. of non update less than max age, to discard lost track
            # 2. no. of update larger than min hit, to eliminate false positive
            if ((trk.time_since_update <= self.max_age) and self.frame_count >= self.min_hits):

                trk.history[img_id] = convert_x_to_bbox(trk.kf.x)
                """
                trk.id starts at 1
                return bbox vel as well 
                """
                ret.append(np.concatenate((d, [trk.id], vel)).reshape(1, -1))

            i -= 1

            # remove dead tracklet when
            # 1. bbox is lost
            # 2. we reach last image
            if (trk.time_since_update > self.max_age) or (img_id == self.last_img_id):
                # check whether we have valid bbox in this tracker
                if len(trk.history) > 0:
                    tracker_id = self.trackers[i].id
                    self.lost_trackers_dict[tracker_id] = self.trackers[i]

                self.trackers.pop(i)

        if(len(ret) > 0):
            return np.concatenate(ret)

        return []