from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import argparse
import os
import os.path as osp
import pandas as pd

from third_party.yolov3.utils import *
from third_party.yolov3.darknet import *

def detect_obj(yolo_model, classes, img_original):
    """
    this function could output none, when there is no
    object detection at all, or an empty list, when there
    is no car detection in detected objects
    :param yolo_model:
    :param classes:
    :param img_original:
    :return:
    """

    CUDA = torch.cuda.is_available()
    batch_size = 1

    # Object Confidence to filter predictions
    confidence = 0.6
    # confidence = 0.3

    # NMS Threshhold
    nms_thresh = 0.4

    #For COCO
    num_classes = 80

    # inp_dim = 416
    inp_dim = int(yolo_model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        yolo_model.cuda()

    #Set the model in evaluation mode
    yolo_model.eval()

    # img_name = '0000000005.png'
    # img_path = file_path + 'img/' + img_name
    # det = file_path + 'det/'
    # det_name = det + img_name

    # img = cv2.imread(img_path)
    img = img_original.copy()
    im_dim_list = [(img.shape[1], img.shape[0])]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    start = time.time()
    img_prep = prep_image(img, inp_dim)

    if CUDA:
        img_prep = img_prep.cuda()

    with torch.no_grad():
        prediction = yolo_model(Variable(img_prep), CUDA)
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)
    end = time.time()

    if type(prediction) == int:
        # print("predicted in {1:6.3f} seconds".format((end - start)/batch_size))
        pass
    else:
        output = prediction
        im_id = 0

        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        # print("predicted in {} seconds".format((end - start)/batch_size))
        # print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        # print("----------------------------------------------------------")

    # The line torch.cuda.synchronize makes sure that CUDA kernel is synchronized with the CPU.
    if CUDA:
        torch.cuda.synchronize()

    try:
        output
    except NameError:
        print ("No detections were made")
        return None, None

    # transform the corner attributes of each bounding box, to the original dimensions of images
    # torch.clamp(input, min, max, out=None)
    # Clamp all elements in input into the range [ min, max ] and return a resulting tensor
    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))
    im_dim_list /= inp_dim
    output[:,1:5] *= im_dim_list

    car_bboxes, bbox_img = output_to_car_bbox(output, classes, img)
    # print('bbox {}'.format(car_bboxes))
    # exit()

    torch.cuda.empty_cache()

    return car_bboxes, bbox_img

if __name__ == '__main__':
    main()
