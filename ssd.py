import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import model_zoo
from pathlib import Path
import os, sys

sys.path.append('../../Object-Detection')
from util import *

#Path variables
dirname = os.path.dirname(__file__)
CFG_PATH =  os.path.join(dirname, '../configs/archs/ssd.cfg')
WEIGHTS_PATH =  os.path.join(dirname, '../Object-Detection/configs/weights/vgg16_weights.pth')
TEST_IMG_PATH =  os.path.join(dirname, '../data/image/samples/dog-cycle-car.png')


### VGG-16 Base
#define VGG base structure, M is max pooling of kernel size 2, stride 2, C is max pooling with cieling enabled
#to deal with cases where input h/w is not divisible by 2
VGG16_CFG =  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512,
            512, 512, 'M', 512, 512, 512, 'M']

def create_modules(cfg, b_norm=False):
        """
        Takes the VGG16 cfg list and parses it into a list of neural network blocks
        to be built. Each block is repersented as a dictionary in the list.
        """
        layer_list = []
        inp_chans = 3
        for layer in cfg:
            if layer == "M":
                layer_list += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(inp_chans, layer, kernel_size=3, padding=1)
                if b_norm:
                    layer_list += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                else:
                    layer_list += [conv2d, nn.ReLU(inplace=True)]
                #update inp channel
                inp_chans = layer
        #extra layers on top of VGG16
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(
                    512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layer_list += [pool5, conv6,
                nn.ReLU(inplace=True), conv7,
                nn.ReLU(inplace=True)]
        return nn.Sequential(*layer_list)

def feature_scaling_layers(cfg):
    """
    Creates the feature scaling layers for the SSD architecture
    """


class VGG(nn.Module):
    def __init__( self, num_classes=1000, init_weights=True, cfg=VGG16_CFG):
        super(VGG, self).__init__()
        self.features = create_modules(cfg)
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG_SDD(nn.Module):
    def __init__(self):
        self.VGG16 = VGG()
def SSD():
    #create model
    model = VGG_SSD()

    #load weights
    trained_weights = torch.load(WEIGHTS_PATH)
    weight_list = list(trained_weights.items())
    model_kvpair = model.state_dict()
    count=0

    for key,value in model_kvpair.items():
        layer_name, weights = weight_list[count]
        model_kvpair[key] = weights
        count+=1

    return model