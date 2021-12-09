import os, sys 
# add path for starmap models
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../third_party/starmap/")
import cv2 
import torch 
from collections import defaultdict
import pickle 

import sem.base_img_proc
import sem.object_tracker
import third_party.yolov3.obj_detection
import third_party.yolov3.utils
import third_party.yolov3.darknet

# define global constants
GEO_KPS_THRESHOLD = 100
FLOW_CHECK_THRESHOLD = .1
RANSAC_THRESHOLD = .5

def load_yolo_model(yolo_path, yolo_weights_path):
    yolo_model = third_party.yolov3.darknet.Darknet(yolo_path + 'cfg/yolov3.cfg')
    yolo_model.load_weights(yolo_weights_path)
    return yolo_model

def load_starmap(starmap_model_path):
    # load starmap models
    starmap_model = torch.load(starmap_model_path)
    return starmap_model

class SemImageProcessor(sem.base_img_proc.ImageProcessor):
    def __init__(self, K, last_img_id, PG, load_detection_flag):
        super().__init__(K)
        # Previous and current images
        self.cam0_prev_img_msg = None
        self.cam0_curr_img_msg = None
        # for geomtric features
        self.cur_geo_feat = []
        self.prev_geo_feat = []
        self.untracked_geo_feat_ids = []
        # for recording tracked features
        self.curr_tracked_feat_id = []
        # for making all features lost at last frame
        self.last_img_id = last_img_id
        # for klt
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # for yolo
        self.yolo_model = load_yolo_model(PG.yolo_path, PG.yolo_weights_path)
        self.classes = third_party.yolov3.utils.load_classes(PG.yolo_path + 'data/coco.names')
        self.yolo_results_path = PG.yolo_results_path
        self.load_detection_flag = load_detection_flag

        # for object tracking
        sort_max_age = 1
        sort_min_hit = 3
        self.my_tracker = sem.object_tracker.ObjectTracker(
            sort_max_age, sort_min_hit, self.last_img_id)

        # for semantic keypoints
        self.starmap_model = load_starmap(PG.starmap_model_path)
        self.starmap_results_path = PG.starmap_results_path
        self.R_starmap_all_dict = {}

        # for storing the bbox info.
        # in each frame
        self.object_frame_exp = defaultdict(list)

