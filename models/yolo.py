import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pathlib import Path
import os, sys

from util import *

#Path variables
dirname = os.path.dirname(__file__)
CFG_PATH =  os.path.join(dirname, '../configs/archs/yolov3.cfg')
WEIGHTS_PATH =  os.path.join(dirname, '../configs/weights/yolov3.weights')
TEST_IMG_PATH =  os.path.join(dirname, '../data/image/samples/dog-cycle-car.png')

#Determine if CUDA is to be used
CUDA = False
if torch.cuda.is_available():
    CUDA = True

def parse_yolo_cfg(cfg_path):
    """
    Takes the yolo v3 config file and parses it into a list of neural network blocks
    to be built. Each block is repersented as a dictionary in the list.

    Arguments:
        cfg_path (string) -- [path to cfg file]
    """
    file = open(str(cfg_path), 'r')
    lines = file.read().split('\n') #list of each line
    lines = [x for x in lines if len(x) > 0] #dont keep empty lines
    lines = [x for x in lines if x[0] != '#'] #dont keep comments
    lines = [x.rstrip().lstrip() for x in lines] #dont keep leading and trailing whitespace

    blocks = []
    block = {}
    for line in lines:
        if line[0] == '[': #beginning of a block in cfg
            if len(block) != 0: #if block contains value of previous block
                blocks.append(block) #add to list
                block = {} #reset value for next block
            block['type'] = line[1:-1].rstrip()

        else: #not in beginning, build up block dict
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    """
    Receives a list of neural network blocks and uses it to return a nn.ModuleList
    containing the blocks.

    Arguments:
        blocks {list} -- list of dicts, where each dict is a neural network block,
                        single block could consist of multiple layers (e.g. conv, bn, upsample)
    """
    #net info has hyperparams and post processing info
    net_info = blocks[0]
    module_list = nn.ModuleList() #to be returned
    prev_filters = 3 #init to rgb channels
    output_filters = [] #track number of output filters for each block to deal with feature map concat in route layer

    #build module list
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if  x['type'] == 'convolutional':
            activation = x['activation']
            filters = int(x['filters'])
            padding = int(x['pad'])
            stride = int(x['stride'])
            kernel_size = int(x['size'])

            #batch norm is either 1 or absent from x
            try:
                batch_norm = int(x['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True

            #must calc the padding
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #add conv layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            #add batch norm
            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("bn_{0}".format(index), bn)

            #add activation (Relu or Linear)
            if activation == "leaky":
                module.add_module('leaky_{0}'.format(index), nn.LeakyReLU(0.1, inplace=True))

        elif x['type'] == 'upsample':
            #pdb.set_trace()
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{}'.format(index), upsample)

        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',') #make into list of layers to retrieve
            start = int(x['layers'][0])

            #see if there is another layer
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            #put indices in terms of current layer
            if start>0: start -=index
            if end>0: end -=index
            route = EmptyLayer() # concat taken care of in nn.module fwd func for darknet
            module.add_module('route_{0}'.format(index), route)

            if end<0: #will be the case if we are concatenating
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index+start]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer() #placeholder
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask] #only keep anchors indicated by mask

            detection = DetectionLayer(anchors) #to be defined later
            module.add_module('Detection_{}'.format(index), detection)

        module_list.append(module) #appends built up block to list of blocks
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):
    """
    Conv net as defined in yolov3.cfg, performs feature extraction
    prior to object detection
    """
    def __init__(self, cfg_path):
        super(Darknet, self).__init__()
        self.blocks = parse_yolo_cfg(CFG_PATH)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA=True):
        modules = self.blocks[1:] #index 0 is net info
        outputs = {} #cache feature maps for route and shortcut
        write = 0 #determines if we have encountered first detection (for concatenation)
        for index, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == 'convolutional' or module_type == "upsample":
                x = self.module_list[index](x) #pass x through current module

            elif module_type == "route":
                layers = module['layers']
                layers = [int(x) for x in layers] #convert to int list

                if layers[0] > 0: layers[0] -= index
                if len(layers) == 1:
                    x = outputs[index+ layers[0]]
                else:
                    if layers[1] > 0: layers[1] -= index
                    map1 = outputs[index+layers[0]]
                    map2 = outputs[index+layers[1]]
                    x = torch.cat((map1,map2), 1) #concat along channel dim

            elif module_type == "shortcut":
                from_ = int(module['from'])
                x = outputs[index-1] + outputs[index+from_]

            elif module_type == "yolo":
                anchors = self.module_list[index][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module['classes'])
                #transform
                x = x.data
                x = predict_transform(x,inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections,x),1)
            outputs[index] = x
        return detections

    def load_weights(self, file_weights):
        """
        Loads weights for conv layers, loads differently based on batch norm presence.

        Arguments:
            file_weights [str] -- path to the file that contains the weights
        """

        file = open(file_weights, 'rb')
        #header info from first 5 values
        header = np.fromfile(file, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        #load weights
        weights = np.fromfile(file, dtype=np.float32)

        #track where in weights array we are
        ptr = 0

        #iterate over layers
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"] #skip header

            if module_type == "convolutional":
                module = self.module_list[i]
                try:
                    batch_norm = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_norm = 0

                conv = module[0]

                if batch_norm:
                    bn = module[1]
                    #num of weights
                    num_bn_biases = bn.bias.numel()
                    #load weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    #transform into correct shape
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #copy into model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    #num of biases and weights
                    num_biases = conv.bias.numel()

                    #load weights, reshape, and copy for bi
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr += num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                #conv weights
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr += num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


test_script = False 
if test_script:
    model = Darknet(CFG_PATH)
    if CUDA: model = model.cuda() #place on gpu
    model.load_weights(WEIGHTS_PATH)
    inp = get_test_input(TEST_IMG_PATH, CUDA)
    pred = model(inp, torch.cuda.is_available())
    print(pred, CUDA)