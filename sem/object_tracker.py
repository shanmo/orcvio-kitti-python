import numpy as np

import sem.kps_tracker
import third_party.sort

class ObjectTracker():
    """
    this class tracks the bbox and semantic keypoints
    """
    def __init__(self, sort_max_age, sort_min_hit, last_img_id):
        # reset tracker id
        third_party.sort.KalmanBoxTracker.count = 1
        self.bbox_tracker = third_party.sort.Sort(sort_max_age, sort_min_hit, last_img_id)
        self.kps_tracker = sem.kps_tracker.KpsTracker()
        self.lost_object_ids = []

def convert_bboxes_list2array(object_bboxes):
    """
    :param object_bboxes: list of detected bbox
    :return: size nx4, bbox in xyxy format
    """
    det = np.zeros((0, 4))
    for bbox in object_bboxes:
        bbox = np.reshape(bbox, (1, -1))
        det = np.concatenate((det, bbox), axis=0)
    return det