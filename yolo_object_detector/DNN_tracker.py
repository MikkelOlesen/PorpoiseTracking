from __future__ import division

from dnn_tracker_needs.models import *
from dnn_tracker_needs.utils.utils import *
from dnn_tracker_needs.utils.datasets import *
from dnn_tracker_needs.utils.augmentations import *
from dnn_tracker_needs.utils.transforms import *
from dnn_tracker_needs.sort import *

import os
import sys
import time
import datetime
import argparse
from time import sleep

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import cv2

# load weights and set defaults
config_path='dnn_tracker_needs/yolov3-custom.cfg'
weights_path='dnn_tracker_needs/yolov3_ckpt_60.pth'
class_path='dnn_tracker_needs/classes.names'
video_path='Videos/20190319_Male_Group_of_porpoises_and_a_calf.MOV'
#video_path='Videos/20190629_Kerteminde_bay_Three_porpoises_Dennis_data.MOV'
#video_path='Videos/20190407_Male_DJI4_Three_porpoises.MOV'
img_size=608
conf_thres=0.9
nms_thres=0.1

# load model and put into eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(config_path, img_size=img_size).to(device)
model.load_state_dict(torch.load(weights_path))
model.cuda()
model.eval()

#load classes
classes = load_classes(class_path)

#load video
capture = cv2.VideoCapture(video_path)

#create tensor
Tensor = torch.cuda.FloatTensor

#create MO tracker
mot_tracker = Sort(6,3,0.3) # max age out of frame, min amount of hits to start tracking, Minimum IOU for match

def detect_image(img): #From https://github.com/cfotache/pytorch_objectdetecttrack object_tracker.py
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        
    return detections[0]


if not capture.isOpened:
    print("Unable to open video")
    exit(0)
while True:
    #timer = cv2.getTickCount()
    ret, frame = capture.read()

    frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)

    if frame is None:
        break

    calcframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(calcframe)
    detections = detect_image(pilimg)

    if detections is not None:
        #rescale Bbounding boxes to original img size
        detections = rescale_boxes(detections, img_size, frame.shape[:2])


        bBoxes_list = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            bBox = [x1, y1, x2, y2, conf]
            bBoxes_list.append(bBox)
        bBoxes = np.array(bBoxes_list)
        #print(bBoxes)

        track_bBoxes_id = mot_tracker.update(bBoxes)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,128),2)
            #cv2.putText(frame,str(classes[int(cls_pred)]),(x2, y2),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),thickness=3)
        
        for bb in track_bBoxes_id:
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]),int(bb[3])), (0,255,0),2)
            cv2.putText(frame,str(int(bb[4])),(int(bb[2]),int(bb[3])),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),thickness=3)
    




    #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.rectangle(frame, (10, 2), (300,100), (255,255,255), -1)
    cv2.putText(frame, str(int(capture.get(cv2.CAP_PROP_POS_FRAMES))), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),thickness=6)
    #cv2.putText(frame, str(int(fps)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),thickness=6)

    
    x = 3
    cv2.namedWindow("Frame",cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Frame', int(frame.shape[1]/x),int(frame.shape[0]/x))
    cv2.imshow("Frame", frame)
    cv2.waitKey(5)
        
