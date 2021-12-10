
import numpy as np
import math
from scipy.linalg import eigh
from collections import defaultdict, namedtuple
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

    def plot_sem_tracks(self, kps_tracker, image_bgr):
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

                thickness = -1
                image_bgr = cv2.circle(image_bgr, (int(x), int(y)), 1, self.colours[part_id]*255, thickness)

                # connect cur kp to next kp
                if part_id in old_points:
                    x_old, y_old = old_points[part_id]
                    thickness = 9
                    image_bgr = cv2.line(image_bgr, (int(x_old), int(y_old)), (int(x), int(y)), self.colours[part_id]*255, thickness)

                # update old points
                old_points[part_id] = [x, y]

        for part_id in old_points:
            kp_2d = old_points[part_id]
            # mark the newest kp detection
            thickness = -1
            image_bgr = cv2.circle(image_bgr, (int(kp_2d[0]), int(kp_2d[1])), 10, self.colours[part_id]*255, thickness)

        return image_bgr

    def plot_semantic_kps(self, bbox_trackers, kps_trackers, image_bgr):
        cur_updated_obj_id_list = []
        for trk in bbox_trackers:
            # get object id
            bbox_track_id = trk[4]
            cur_updated_obj_id_list.append(bbox_track_id)

        for object_id, trk in kps_trackers.trackers.items():
            # only plot objects visible in cur frame
            if object_id not in cur_updated_obj_id_list:
                continue
            image_bgr = self.plot_sem_tracks(trk, image_bgr)
        
        return image_bgr

    def plot_all(self, image_bgr, bbox_trackers, kps_trackers, img_id):
        # plot bbox
        image_bgr = self.plot_bbox(bbox_trackers, image_bgr, img_id)
        # plot semantic keypoints
        image_bgr = self.plot_semantic_kps(bbox_trackers, kps_trackers, image_bgr)
        return image_bgr