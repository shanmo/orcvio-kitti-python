# This file will contain the code that creates the YOLO network
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from third_party.yolov3.utils import *

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    #Resize to the input dimension
    img = cv2.resize(img, (416, 416))
    # BGR -> RGB | H W C -> C H W
    img_ = img[:, :, ::-1].transpose((2,0,1))
    #Add a channel at 0 (for batch) | Normalise
    img_ = img_[np.newaxis, :, :, :]/255.0
    #Convert to float
    img_ = torch.from_numpy(img_).float()
    # Convert to Variable
    img_ = Variable(img_)
    return img_


def parse_cfg(cfgfile):
    """
    takes a cfg file

    returns a list of blocks, each block describes a block in the cnn. block is represented
    as a dict in the list
    """
    # saving the content of the cfg file in a list of strings
    file = open(cfgfile, 'r')
    # store the lines in a list
    lines = file.read().split('\n')
    # get rid of empty lines
    lines = [x for x in lines if len(x) > 0]
    # get rid of comments
    lines = [x for x in lines if x[0] != '#']
    # get rid of spaces
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        # this marks the start of a new block
        if line[0] == "[":
            #if block is not empty
            if len(block) != 0:
                # add block dict to blocks list
                blocks.append(block)
                # clear the block
                block = {}
            # eg net, line is [net]
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            # del the spaces before and after =
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append to module_list

        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #Check the activation.
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        # initialize the network with members, blocks, net_info and module_list
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA=False):
        # 1st ele self.blocks is a net block which isn't a part of the forward pass
        modules = self.blocks[1:]
        #We cache the outputs for the route layer
        # Since route and shortcut layers need output maps from previous layers, we cache the output feature maps of every layer in a dict outputs
        outputs = {}
        # used to indicate whether we have encountered the first detection or not
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])
            # Convolutional and Upsample Layers
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            # Route Layer / Shortcut Layer
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    # TODO why not outputs[layers[0]]??
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    # the case in which we have to concatenate two feature maps we use the torch.cat function with the second argument as 1. This is because we want to concatenate the feature maps along the depth
                    # b, c, h, w, the depth corresponding the the channel dimension
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                # adds a feature maps of a previous layer to those of layer behind
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int(self.net_info["height"])
                #Get the number of classes
                num_classes = int(module["classes"])
                #Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                #if no collector has been intialised.
                if not write:
                    detections = x
                    write = 1
                else:
                    # 1st is batch! so 2nd one is row!
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        # The row of each of this table represents a bounding box. (4 bbox attributes, 1 objectness score, and 80 class scores)
        return detections

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            #If module_type is convolutional load weights
            #Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]


                if (batch_normalize):
                    bn = model[1]

                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #Number of biases
                    num_biases = conv.bias.numel()

                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


if __name__ == '__main__':
    file_path = '/Users/moshan/Documents/PhD/research/yolo_from_scratch/cfg/yolov3.cfg'
    model = Darknet(file_path)
    model.load_weights("yolov3.weights")
    inp = get_test_input()
    pred = model(inp)
    # print(pred)
