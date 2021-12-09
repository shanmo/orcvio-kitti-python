# contain the code for various helper functions
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import pickle as pkl
import random
import os, sys

# load colors
# dir_path = os.path.abspath(os.path.join(__file__ ,"../../"))
# yolo_path = dir_path + '/yolov3/'
# colors = pkl.load(open(yolo_path + "pallete", "rb"))

"""bbox rejection"""
"""reject small cars since they may be false alarms due to occlusion"""
bbox_thresh_min = 30
"""reject large bbox"""
bbox_thresh_max = 1e3

ratio_thresh = 0.8

def write(x, classes, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    # color = random.choice(colors)
    """use green for car"""
    color = (0, 255, 0)
    label = "{0}".format(classes[cls])
    # cv2.rectangle(img, c1, c2,color, 1)
    """make bounding box larger"""
    cv2.rectangle(img, c1, c2, color, 3)
    # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 3 , 3)[0]
    # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # cv2.rectangle(img, c1, c2, color, -1)
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

# refer to https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/issues/4
# There is a variable output, a 2-d tensor which contains the details of each detection per row.
# the column number 0 represents the index of the image in imlist, the columns 1-4 represent the corner co-ordinates of that bounding box in form (x1, y1, x2, y2) and the last column is the object class of that detection
def output_to_car_bbox(output, classes, img):
    # for coco names
    CAR_CLASS = 2
    TRUCK_CLASS = 7
    BUS_CLASS = 5

    # for i in range(output.size()[0]):
    #     x = output[i, :]
    #     bbox_img = write(x, img)

    bboxes = []
    bbox_img = None
    for i in range(output.size()[0]):
        detection = output[i, :]
        object_class = detection[7]
        # bus, truck detection not good
        # if object_class == CAR_CLASS or object_class == TRUCK_CLASS or object_class == BUS_CLASS:
        if object_class == CAR_CLASS:
            bbox_img = write(detection, classes, img)
            # convert tensor to array
            bbox = np.array(detection[1:5])
            """
            bbox is x1y1x2y2 format 
            """
            # To further reduce false positive, we include thresholds for bounding box width, height, and height-to-width ratio.
            # hight = y2 - y1
            # width = x2 - x1
            box_h = bbox[3] - bbox[1]
            box_w = bbox[2] - bbox[0]
            ratio = box_h / (box_w + 0.01)
            if ((ratio < ratio_thresh) and (box_h > bbox_thresh_min) and (box_w > bbox_thresh_min) and \
                (box_w < bbox_thresh_max)):
                bboxes.append(bbox)
            else:
                # print('wrong ratio {} or wrong size {} {}'.format(ratio, box_h, box_w))
                pass
            # bboxes.append(bbox)
    # bboxes is a list of array
    return bboxes, bbox_img

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = False):
    """
    predict_transform function takes an detection feature map and turns it into a 2-D tensor, where each row of the tensor corresponds to attributes of a bounding box
    """
    batch_size = prediction.size(0)
    # inp_dim (input image dimension)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # The dimensions of the anchors are in accordance to the height and width attributes of the net block. These attributes describe the dimensions of the input image, which is larger (by a factor of stride) than the detection map. Therefore, we must divide the anchors by the stride of the detection feature map
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Add the grid offsets to the center cordinates prediction.
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)
    # change x, y offsets to column vectors
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    # offset.repeat(1, num_anchors)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # Apply the anchors to the dimensions of the bounding box
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    # TODO why....
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    # Apply sigmoid activation to the the class scores
    prediction[:, :, 5:5 + num_classes] = torch.sigmoid((prediction[:, :, 5:5 + num_classes]))
    # The last thing we want to do here, is to resize the detections map to the size of the input image. The bounding box attributes here are sized according to the feature map (say, 13 x 13). If the input image was 416 x 416, we multiply the attributes by 32, or the stride variable
    prediction[:, :, :4] *= stride

    return prediction

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    # TODO why???
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """
    The first input is the bounding box row that is indexed by the the variable i in the loop.
    Second input to bbox_iou is a tensor of multiple rows of bounding boxes. The output of the function bbox_iou is a tensor containing IoUs of the bounding box represented by the first input with each of the bounding boxes present in the second input.
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    #Intersection area
    inter_area = (inter_rect_x2 - inter_rect_x1 + 1)*(inter_rect_y2 - inter_rect_y1 + 1)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    """
    The functions takes as as input the prediction, confidence (objectness score threshold), num_classes (80, in our case) and nms_conf (the NMS IoU threshold)
    The function write_results outputs a tensor of shape D x 8. Here D is the true detections in all of images, each represented by a row. Each detections has 8 attributes, namely, index of the image in the batch to which the detection belongs to, 4 corner coordinates, objectness score, the score of class with maximum confidence, and the index of that class.
    """
    # for each of the bounding box having a objectness score below a threshold, we set the values of it's every attribute (entire row representing the bounding box) to zero
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    # we transform the (center x, center y, height, width) attributes of our boxes, to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    # loop over the first dimension of prediction (containing indexes of images in a batch)
    batch_size = prediction.size(0)
    # write flag is used to indicate that we haven't initialized output, a tensor we will use to collect true detections across the entire batch
    write = False
    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
        #confidence threshholding
        #NMS
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        # Remember we had set the bounding box rows having a object confidence less than the threshold to zero? Let's get rid of them
        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        # The try-except block is there to handle situations where we get no detections. In that case, we use continue to skip the rest of the loop body for this image
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        #For PyTorch 0.4 compatibility
        #Since the above code with not raise exception for no detection
        #as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue

        # get the classes detected in a an image
        # -1 index holds the class index
        img_classes = unique(image_pred_[:,-1])

        for cls in img_classes:
            #perform NMS
            #get the detections with one particular class
            # TODO what???
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            #Number of detections
            # TODO why max gives many detections ????
            idx = image_pred_class.size(0)
            # perform NMS
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at
                # even if one value is removed from image_pred_class, we cannot have idx iterations
                try:
                    # In the body of the loop, the following lines gives the IoU of box, indexed by i with all the bounding boxes having indices higher than i
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    # there may be no detections
    try:
        return output
    except:
        return 0

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def prep_image(img, inp_dim):
    """
    OpenCV loads an image as an numpy array, with BGR as the order of the color channels. PyTorch's image input format is (Batches x Channels x Height x Width), with the channel order being RGB.
    """
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img