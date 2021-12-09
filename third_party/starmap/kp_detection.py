import torch
import cv2
import numpy as np

import math
import torch.nn as nn

from third_party.starmap.hmParser import parseHeatmap
from third_party.starmap.horn87 import horn87
from third_party.starmap.config import *

import third_party.starmap.obj_structure

## check whether we can use GPU
CUDA = torch.cuda.is_available()

def rotmat_2D_from_angle(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]])

def NewCrop(img, max_side, desired_side, rot = 0,
         points = np.zeros((0, 2)),
         return_points = False):
    """
    Scale and Crop and fit the image into the desired_side x desired_side
    old crop function is confusing.
    """

    # resize the image only if we have reduce the size by more than half
    resized_img = cv2.resize(img,
                            ((img.shape[1] * desired_side // max_side),
                             (img.shape[0] * desired_side // max_side)))

    # Cropping begins here
    # The image rectangle clockwise
    rect_resized = np.array([
    [0, 0],
    [resized_img.shape[1], 0],
    [resized_img.shape[1], resized_img.shape[0]],
    [0, resized_img.shape[0]],
    ])

    # Project the rectangle from source image to target image
    # TODO account for rotation
    target_center = np.array([[desired_side, desired_side ]]) / 2
    resized_img_center = rect_resized.mean(axis=-2, keepdims=True)
    R = rotmat_2D_from_angle(rot)

    rect_target = (
      (R @ (rect_resized - resized_img_center).T).T
      + target_center)

    img_center = np.array([[img.shape[1], img.shape[0]]]) / 2
    target_points = (((desired_side / max_side) * R @ (points - img_center).T).T
                   + target_center)

    # Find the range of the rectangle
    mins_target = np.int64(np.ceil(np.maximum(np.min(rect_target, axis=-2),
                         [0, 0])))
    maxs_target = np.int64(np.floor(np.minimum(np.max(rect_target, axis=-2),
                                             [desired_side, desired_side])))

    new_img_shape = ((desired_side, desired_side, img.shape[2])
                   if img.ndim > 2
                   else (desired_side, desired_side))
    new_img = np.zeros(new_img_shape, dtype=img.dtype)

    # Copy the cropped region from original image to target image
    new_img[mins_target[1]:maxs_target[1],
          mins_target[0]:maxs_target[0],
          ...] = resized_img[0:maxs_target[1]-mins_target[1],
                             0:maxs_target[0]-mins_target[0], ...]

    # newImg = new_img.reshape((-1, desired_side, desired_side))
    newImg = new_img.transpose(2, 0, 1).astype(np.float32)

    return ((newImg, target_points)
          if return_points
          else newImg)


def get_original_pts(coords_list, img_center, max_side, desired_side):

    # Transform predictions back to original coordinate space

    rot = 0
    R = rotmat_2D_from_angle(rot)

    target_center = np.array([[desired_side, desired_side]]) / 2

    kp_num = len(coords_list)
    new_coords = coords_list.copy()

    for k in range(kp_num):

        target_points = coords_list[k]
        new_point = max_side / desired_side * (R.T @ (target_points -
                        target_center).T).T + img_center

        new_coords[k][0] = new_point[0, 1]
        new_coords[k][1] = new_point[0, 0]

    return new_coords

def set_bn_to_eval(layer):

    if type(layer) == nn.modules.batchnorm.BatchNorm2d:
        # print(type(layer))
        layer.training = False

def get_R_starmap(hm, ps):
    """
    get R from starmap output
    :param hm:
    :param ps:
    :return:
    """

    canonical, pred, score = [], [], []

    for k in range(len(ps[0])):

        x, y, z = ((hm[0, 1:4, ps[0][k], ps[1][k]] + 0.5) * 64).astype(np.int32)
        dep = ((hm[0, 4, ps[0][k], ps[1][k]] + 0.5) * 64).astype(np.int32)
        canonical.append([x, y, z])

        dep = ((hm[0, 4, ps[0][k], ps[1][k]] + 0.5) * 64).astype(np.int32)
        pred.append([ps[1][k], 64 - dep, 64 - ps[0][k]])
        score.append(hm[0, 0, ps[0][k], ps[1][k]])

    pred = np.array(pred).astype(np.float32)
    canonical = np.array(canonical).astype(np.float32)

    pointS = canonical * 1.0 / 64
    pointT = pred * 1.0 / 64
    R, t, s = horn87(pointS.transpose(), pointT.transpose(), score)

    return R

def get_kps_and_R(model, input_var, heat_thresh, img):
    """
    detect kps using starmap no dropout model
    :param model:
    :param input_var:
    :param heat_thresh:
    :param img:
    :return:
    """

    """
    show kp detection 
    on crop img 
    """
    cv_plot_flag = False

    all_kps_dict = {}

    output = model(input_var)
    hm = output[-1].data.cpu().numpy()

    ps = parseHeatmap(hm[0], heat_thresh)

    """obtain rotation"""
    R_starmap = get_R_starmap(hm, ps)

    kp_num = len(ps[0])

    for k in range(kp_num):
        """cam view feat"""
        # 64 = 64
        x, y, z = ((hm[0, 1:4, ps[0][k], ps[1][k]] + 0.5) * 64).astype(np.int32)
        cam_view_feat = np.array([x, y, z])
        part_id, part_name = third_party.starmap.obj_structure.find_semantic_part(
                                cam_view_feat)

        kp = np.array([[ps[1][k] * 4, ps[0][k] * 4]]).T

        try:
            all_kps_dict[part_id] = np.hstack((all_kps_dict[part_id], kp))
        except:
            all_kps_dict[part_id] = kp

    pos_list = []
    labels_list = []

    if len(all_kps_dict) != 0:

        """convert color to grayscale"""
        img = np.transpose(img, (1, 2, 0))
        img_bgr = img.copy()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i, key in enumerate(all_kps_dict):
            kp = all_kps_dict[key]

            """
            there might be multiple detections for the same 
            keypoint, so we need to use mean 
            """
            kp_mean = np.mean(kp, axis=1)

            cv2.circle(img_bgr, (int(kp_mean[0]), int(kp_mean[1])), 10, (0, 0, 255), -1)
            cv2.putText(img_bgr, str(key),
                (int(kp_mean[0]), int(kp_mean[1])), cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 3)

            # pos_list could be list of lists?
            pos_list.append(kp_mean)

            labels_list.append(key)

        if cv_plot_flag:
            cv2.imshow("cv2Im", img_bgr)
            cv2.waitKey(0)

    return pos_list, labels_list, R_starmap



def detect_kp(starmap_model, original_img):
    """
    detect kps using starmap
    with MC dropout
    :param starmap_model:
    :param original_img:
    :return:
    """

    height = original_img.shape[0]
    width = original_img.shape[1]
    if height <= 0 or width <= 0:
        original_ps = []
        labels_list = []
        R_starmap = []
        return original_ps, labels_list, R_starmap


    img = original_img.copy()

    # crop change h, w, c to c, 256, 256
    s = max(original_img.shape[0], original_img.shape[1])
    img = NewCrop(img, s, inputRes) / 256.

    input = torch.from_numpy(img.copy()).float()

    # change to b, c, h, w
    input = input.view(1, input.size(0), input.size(1), input.size(2))
    input_var = torch.autograd.Variable(input).float()

    if CUDA:
        starmap_model.cuda()
        input_var = input_var.cuda()

    heat_thresh = 0.01

    pos_list, labels_list, R_starmap = \
        get_kps_and_R(starmap_model, input_var, heat_thresh, img)

    c = np.array([original_img.shape[1]/2., original_img.shape[0]/2.])
    original_ps = get_original_pts(pos_list, c, s, inputRes)

    return original_ps, labels_list, R_starmap
