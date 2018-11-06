from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import pdb

def bbox_iou(box1,box2):
    """
    returns the iou between 2 boxes, or one box and a series of boxes
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1,min=0) *torch.clamp(inter_rect_y2-inter_rect_y1+1, min=0)
    b1_area = (b1_x2-b1_x1+1) * (b1_y2-b1_y1+1)
    b2_area = (b2_x2-b2_x1+1) * (b2_y2-b2_y1+1)
    union_area = b1_area + b2_area - inter_area
    iou = inter_area/union_area
    return iou

def get_test_input(path, CUDA, inp_dim):
    """
    Reads in a single image using cv2 and performs some preprocessing
    to prepare it for the pytorch model.

    path (str) : path to input image
    CUDA (bool) : for gpu acceleration
    """
    img = cv2.imread(path)
    img = process_image(img, inp_dim)
    img = Variable(img)
    if CUDA:
        img = img.cuda()
    return img


def load_classes(namesfile):
    """
    Returns a dictionary that maps the index of a class to its name
    """
    file = open(namesfile)
    names = file.read().split("\n")[:-1]
    return names


def predict_transform(pred, inp_dim, anchors, num_classes, CUDA=True):
    """
    Takes a 3D detection feature map and turns it into a 2D tensor,
    where each row would be a single bounding boxes attributes (e.g
    first three rows would be three bounding boxes at cell 0,0, then 0,1). This allows
    for computations to be run on all possible concatenated feature maps in one go,
    by turning them all 2D. Otherwise concatenating is not possible for feature maps of different sizes.
    """
    batch_sz = pred.size(0)
    stride = inp_dim//pred.size(2)
    grid_size = inp_dim//stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    #fix anchors with stride so dims are int term by detection map not input
    anchors = [(x[0]/stride, x[1]/stride) for x in anchors]

    #reshape preds
    pred = pred.view(batch_sz, bbox_attrs*num_anchors, grid_size*grid_size)
    pred = pred.transpose(1,2).contiguous()
    pred = pred.view(batch_sz, grid_size*grid_size*num_anchors, bbox_attrs)

    #sigmoid the bbox center coords and objectness score
    pred[:,:,0] = torch.sigmoid(pred[:,:,0])
    pred[:,:,1] = torch.sigmoid(pred[:,:,1])
    pred[:,:,4] = torch.sigmoid(pred[:,:,4])

    #grid offsets added to center coords
    grid = np.arange(grid_size)
    xx, yy = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(xx).view(-1,1)
    y_offset = torch.FloatTensor(yy).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        pred = pred.cuda()

    x_y_offset = torch.cat((x_offset,y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    pred[:,:,:2] += x_y_offset

    #applying log space transform on output and multiply with anchor for width and height
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0) #ensures anchors repeated for whole grid, for each training sample (unsqueeze)
    pred[:,:,2:4] = torch.exp(pred[:,:,2:4]) * anchors

    #sigmoid to class scores
    pred[:,:,5:5+num_classes] = torch.sigmoid(pred[:,:,5:5+num_classes])

    #bbox coords back in terms of the input image instead of the feature map
    pred[:,:,:4] *= stride

    return pred

def process_image(img, inp_dim):
    """
    Prepare an image for inputting to network in Pytorchs correct format
    """
    #resize to network input dim
    img = (scale_image(img, (inp_dim,inp_dim)))
    #change order to rgb
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    #convert to numpy float array, normalize, add an axis to build batch
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def process_results(pred, conf, num_classes, nms_conf=0.4):
    conf_mask = (pred[:,:,4]>conf).float().unsqueeze(2)
    pred = pred * conf_mask
    #transform coords from center x,y,width,height to top lefts and bottom right coords of box
    box_corner = pred.new(pred.shape)
    box_corner[:,:,0] = (pred[:,:,0] - pred[:,:,2]/2)
    box_corner[:,:,1] = (pred[:,:,1] - pred[:,:,3]/2)
    box_corner[:,:,2] = (pred[:,:,0] + pred[:,:,2]/2)
    box_corner[:,:,3] = (pred[:,:,1] + pred[:,:,3]/2)
    pred[:,:,:4] = box_corner[:,:,:4]

    batch_sz = pred.size(0)
    write = False

    #Performed on a single image at a time as number of true detections can vary.
    for ind in range(batch_sz):
        #remove the 80 scores and add index of class with max prediction and class score
        img_pred = pred[ind]
        max_conf, max_conf_ind = torch.max(img_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_ind = max_conf_ind.float().unsqueeze(1)
        seq = (img_pred[:,:5], max_conf, max_conf_ind)
        img_pred = torch.cat(seq, 1)

        #threshold - get rid of zero rows
        non_zero_ind =  (torch.nonzero(img_pred[:,4]))
        try:
            img_pred_ = img_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        # pytorch 4.0 compatibility, above block wont raise exception as scalars are supported
        if img_pred_.shape[0] == 0:
            continue

        #Get the index of classes detected in the image
        img_classes = unique(img_pred_[:,-1])
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = img_pred_*(img_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            img_pred_class = img_pred_[class_mask_ind].view(-1,7)

            #sort the detections such that the entry with the maximum obj conf is at the top
            conf_sort_index = torch.sort(img_pred_class[:,4], descending = True )[1]
            img_pred_class = img_pred_class[conf_sort_index]
            idx = img_pred_class.size(0)   #Number of detections

            #NMS
            for i in range(idx):
            #Get the IOUs of all boxes that come after the one we are looking at
                try:
                    ious = bbox_iou(img_pred_class[i].unsqueeze(0), img_pred_class[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break

                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                img_pred_class[i+1:] *= iou_mask

                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(img_pred_class[:,4]).squeeze()
                img_pred_class = img_pred_class[non_zero_ind].view(-1,7)

            batch_ind = img_pred_class.new(img_pred_class.size(0), 1).fill_(ind)
            #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, img_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    try:
        return output
    except:
        return 0

def scale_image(img, inp_dim):
    """
    resize image by adding padding such that the aspect ratio does not change
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas

def unique(tensor):
    """
    Uses numpy's built in unique method to return a copy of the tensor giving
    only the unique classes present.
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res
