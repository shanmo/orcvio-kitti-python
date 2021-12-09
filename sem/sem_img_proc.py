import os, sys 
# add path for starmap models
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../third_party/starmap/")
import cv2 
import torch 
from collections import defaultdict
import pickle 
import numpy as np 
import random

import sem.base_img_proc
import sem.object_tracker
import third_party.yolov3.obj_detection
import third_party.yolov3.utils
import third_party.yolov3.darknet
import third_party.starmap.kp_detection

def load_yolo_model(yolo_path, yolo_weights_path):
    yolo_model = third_party.yolov3.darknet.Darknet(yolo_path + 'cfg/yolov3.cfg')
    yolo_model.load_weights(yolo_weights_path)
    return yolo_model

def load_starmap(starmap_model_path):
    # load starmap models
    starmap_model = torch.load(starmap_model_path)
    return starmap_model

def save_detection_yolo(img_id, base_yolo_results_path, object_bboxes):
    """
    save the detected bbox
    :param img_id: the frame id
    :param base_yolo_results_path: the path to store results
    :param object_bboxes: detected bbox
    """
    yolo_results_path = base_yolo_results_path + str(img_id)
    if not os.path.exists(yolo_results_path):
        os.makedirs(yolo_results_path)
    yolo_results_path += '/object_bboxes.pickle'
    with open(yolo_results_path, 'wb') as handle:
        pickle.dump(object_bboxes, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_detection_yolo(img_id, base_yolo_results_path):
    """
    load detected bbox
    :param img_id: frame id
    :param base_yolo_results_path: directory of results
    :return: detected bbox
    """
    yolo_results_path = base_yolo_results_path + str(img_id)
    yolo_results_path += '/object_bboxes.pickle'
    with open(yolo_results_path, 'rb') as handle:
        object_bboxes = pickle.load(handle)
    return object_bboxes

def detect_object_obs(img_bgr, load_detection_flag, yolo_model, classes,
                      img_id, yolo_results_path):
    """
    detect objects using yolo then
    detect kps using starmap
    :param img_bgr: colored image
    :param load_detection_flag: a flag to determine whether we want to detect or load
    :param yolo_model: trained yolo model
    :param classes: coco classes
    :param img_id: frame id
    :param yolo_results_path: directory for results
    :return: a flag that indicates whether there are bbox, and if yes the detected bbox
    """
    bbox_detected_flag = True
    # object_bboxes contains all bboxes in one frame, unordered
    # bbox is x1y1x2y2 format from yolo
    if load_detection_flag is False:
        object_bboxes, bbox_img = third_party.yolov3.obj_detection.detect_obj(
            yolo_model, classes, img_bgr)
        # this function could output none, when there is no
        # object detection at all, or an empty list, when there
        # is no car detection in detected objects
        if object_bboxes is None:
            object_bboxes = []
        if not object_bboxes:
            bbox_detected_flag = False
        save_detection_yolo(img_id, yolo_results_path, object_bboxes)
    else:
        object_bboxes = load_detection_yolo(img_id, yolo_results_path)
        if not object_bboxes:
            bbox_detected_flag = False
    return object_bboxes, bbox_detected_flag

def convert_kp_to_cv_kp(kps):
    """
    convert kps to cv kp
    """
    cv_kps = []
    for kp in kps:
        cv_kps.append(cv2.KeyPoint(kp[0], kp[1], 5, class_id=0))
    return cv_kps

def load_detection_starmap(img_id, base_starmap_results_path):
    starmap_results_path = base_starmap_results_path + str(img_id)
    kp_ids_path = starmap_results_path
    kp_ids_path += '/sem_kp_object_ids.pickle'

    with open(kp_ids_path, 'rb') as handle:
        sem_kp_object_ids = pickle.load(handle)

    kps_path = starmap_results_path
    kps_path += '/semantic_kps.pickle'

    with open(kps_path, 'rb') as handle:
        kps = pickle.load(handle)
    semantic_kps = convert_kp_to_cv_kp(kps)

    kp_labels_path = starmap_results_path
    kp_labels_path += '/kp_labels.pickle'
    with open(kp_labels_path, 'rb') as handle:
        kp_labels = pickle.load(handle)
    R_starmap_path = starmap_results_path
    R_starmap_path += '/R_starmap_all_dict.pickle'
    with open(R_starmap_path, 'rb') as handle:
        R_starmap_all_dict = pickle.load(handle)
    return sem_kp_object_ids, semantic_kps, kp_labels, R_starmap_all_dict

def crop_image(bbox_det, original_img, w, h):
    """
    crop image using given bbox
    :param bbox_det: bbox in xyxy format
    :param original_img: original image
    :return: cropped image, top left corner of cropped region
    """
    img = original_img.copy()
    # Crop image
    x1, y1, x2, y2 = bbox_det[0], bbox_det[1], bbox_det[2], bbox_det[3]
    # constrain the bbox by image boundary
    x1 = np.clip(x1, 0, w)
    x1 = int(x1)
    x2 = np.clip(x2, 0, w)
    x2 = int(x2)

    y1 = np.clip(y1, 0, h)
    y1 = int(y1)
    y2 = np.clip(y2, 0, h)
    y2 = int(y2)
    im_crop = img[y1:y2, x1:x2]
    return im_crop, x1, y1

def convert_cv_kp_to_kp(semantic_kps):
    """
    convert cv2 kps to kps
    :param semantic_kps: a list of kps in opencv keypoint format
    :return: a list of tuples of kps
    """
    kps = []
    for cv_kp in semantic_kps:
        kps.append((cv_kp.pt[0], cv_kp.pt[1]))
    return kps

def save_detection_starmap(img_id, base_starmap_results_path, sem_kp_object_ids,
                semantic_kps, kp_labels, R_starmap_all_dict):
    starmap_results_path = base_starmap_results_path + str(img_id)
    if not os.path.exists(starmap_results_path):
        os.makedirs(starmap_results_path)
    kp_ids_path = starmap_results_path
    kp_ids_path += '/sem_kp_object_ids.pickle'

    with open(kp_ids_path, 'wb') as handle:
        pickle.dump(sem_kp_object_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    kps_path = starmap_results_path
    kps_path += '/semantic_kps.pickle'
    with open(kps_path, 'wb') as handle:
        kps = convert_cv_kp_to_kp(semantic_kps)
        pickle.dump(kps, handle, protocol=pickle.HIGHEST_PROTOCOL)
    kp_labels_path = starmap_results_path
    kp_labels_path += '/kp_labels.pickle'
    with open(kp_labels_path, 'wb') as handle:
        pickle.dump(kp_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    R_starmap_path = starmap_results_path
    R_starmap_path += '/R_starmap_all_dict.pickle'
    with open(R_starmap_path, 'wb') as handle:
        pickle.dump(R_starmap_all_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def detect_semantic_kps(img_original, starmap_model, bbox_trackers, load_detection_flag,
                        starmap_results_path, img_id, img_shape):
    """
    detect kps using starmap
    :param img_original: original colored (bgr) image
    :param starmap_model: trained starmap model
    :param bbox_trackers: tracked bbox
    :param a flag that indicates whether we want to save or load the results
    :param the directory where we store the starmap detections
    :param index of current image
    :return: detected semantic keypoints
    """
    if load_detection_flag:
        sem_kp_object_ids, semantic_kps, kp_labels, R_starmap_all_dict = \
            load_detection_starmap(img_id, starmap_results_path)
        return sem_kp_object_ids, semantic_kps, kp_labels, R_starmap_all_dict

    kps_all = []
    bbox_id_all = []
    kp_label_all = []
    R_starmap_all_dict = {}

    for bbox_det in bbox_trackers:
        # get object id
        bbox_track_id = bbox_det[4]
        im_crop, x1, y1 = crop_image(bbox_det, img_original, img_shape[0], img_shape[1])
        ps_list, label_list, R_starmap = third_party.starmap.kp_detection.detect_kp(
                starmap_model, im_crop)

        # when there is no keypoints detection
        if len(ps_list) == 0:
            continue

        # record part labels
        for l in label_list:
            kp_label_all.append(l)

        R_starmap_all_dict[bbox_track_id] = R_starmap
        # check whether we have valid kp detections
        kp_num = len(ps_list)
        for k in range(kp_num):
            # x to the right, y to the down
            # note x, y order
            ps_list[k][1] += int(x1)
            ps_list[k][0] += int(y1)
            kp = cv2.KeyPoint(ps_list[k][1], ps_list[k][0], 5, class_id=0)
            kps_all.append(kp)
            bbox_id_all.append(bbox_track_id)

    save_detection_starmap(img_id, starmap_results_path, bbox_id_all,
                kps_all, kp_label_all, R_starmap_all_dict)
    return bbox_id_all, kps_all, kp_label_all, R_starmap_all_dict

class SemImageProcessor(sem.base_img_proc.ImageProcessor):
    def __init__(self, K, img_shape, last_img_id, PG, load_detection_flag):
        super().__init__(K)
        self.img_shape = img_shape
        # Previous and current images
        self.cam0_prev_img_msg = None
        self.cam0_curr_img_msg = None
        # for making all features lost at last frame
        self.last_img_id = last_img_id
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

    def add_new_frame(self, img_msg = None):
        """
        detect new bbox
        track the detected bbox
        """
        self.cam0_curr_img_msg = img_msg
        # detect and track object bbox
        object_bboxes = self.detec_and_track_bbox()
        self.save_bbox_info()
        # detect and track semantic keypoints
        self.detect_and_track_sem_kps()
        # for tracking next frame
        self.cam0_prev_img_msg = self.cam0_curr_img_msg

    def detec_and_track_bbox(self):
        object_bboxes, bbox_detected_flag = detect_object_obs(self.cam0_curr_img_msg.img_bgr, self.load_detection_flag,
                    self.yolo_model, self.classes, self.img_id, self.yolo_results_path)
        # return an empty list if there is no bbox
        if not bbox_detected_flag:
            self.bbox_trackers = []
        det = sem.object_tracker.convert_bboxes_list2array(object_bboxes)
        # self.bbox_trackers stores bbox trackers that are good
        self.bbox_trackers = self.my_tracker.bbox_tracker.update(det, self.img_id)
        return object_bboxes

    def save_bbox_info(self):
        for i in range(np.shape(self.bbox_trackers)[0]):
            trk = self.bbox_trackers[i]
            object_id = trk[4]
            bbox = trk[0:4]
            object_bbox_info = [object_id, *bbox]
            # store all bbox info. in each frame in a list
            self.object_frame_exp[self.img_id].append(object_bbox_info)

    def detect_and_track_sem_kps(self):
        """
        detects and tracks the semantic keypoints given the tracked bbox
        :return:
        """
        sem_kp_object_ids, semantic_kps, kp_labels, R_starmap_all_dict = \
            detect_semantic_kps(self.cam0_curr_img_msg.img_bgr, self.starmap_model, self.bbox_trackers,
                    self.load_detection_flag, self.starmap_results_path, self.img_id, self.img_shape)
        self.R_starmap_all_dict[self.img_id] = R_starmap_all_dict
        # track sem kps
        self.my_tracker.kps_tracker.update(self.bbox_trackers, sem_kp_object_ids, semantic_kps, kp_labels, self.img_id)

    def publish_features(self):
        """
        publish features that are lost
        """
        # observations to be published
        obs_publish_dict = {}
        # publish lost object features
        for object_id, bbox_tracker in self.my_tracker.bbox_tracker.lost_trackers_dict.items():
            # python does not support duplicated keys
            # this is a workaround
            object_dict_id = object_id
            if object_id in self.obs_all_dict:
                object_dict_id = object_id + random.randint(1, 1000)
            self.obs_all_dict[object_dict_id] = {}
            obs_type = 'object'
            img_id_list = []
            zs = np.zeros((0, 12, 2))
            zb = np.zeros((0, 4))
            R0 = np.zeros((0, 3, 3))
            self.obs_all_dict[object_dict_id] = {'type': obs_type, 'img_id': img_id_list, 'zs': zs, 'zb': zb, 'R0': R0}
            for img_id, bbox in bbox_tracker.history.items():
                # we do not have a valid detection when there is no starmap rotation
                if object_id not in self.R_starmap_all_dict[img_id]:
                    pass
                else:
                    self.add_img_id_to_obs_dict(object_dict_id, img_id)
                    bbox = sem.base_img_proc.normalize_bbox(self.K, bbox)
                    self.add_zb_to_obs_dict(object_dict_id, bbox)

            # we may not have valid keypoints detection for a bbox
            if object_id not in self.my_tracker.kps_tracker.trackers:
                continue
            else:
                kp_tracker = self.my_tracker.kps_tracker.trackers[object_id]

            for img_id, all_kps in kp_tracker.history.items():
                # we do not have a valid detection when there is no starmap rotation
                if object_id not in self.R_starmap_all_dict[img_id]:
                    pass
                else:
                    zs = sem.base_img_proc.normalize_pixel(self.K, all_kps)
                    self.add_zs_to_obs_dict(object_dict_id, zs)
                    R_starmap = self.R_starmap_all_dict[img_id][object_id]
                    self.add_R0_to_obs_dict(object_dict_id, R_starmap)

            obs_publish_dict[object_dict_id] = self.obs_all_dict[object_dict_id]

        # reset track to del
        self.my_tracker.bbox_tracker.lost_trackers_dict = {}
        return obs_publish_dict