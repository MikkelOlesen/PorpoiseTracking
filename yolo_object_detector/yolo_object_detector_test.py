from __future__ import division

from yolo_object_detector_test_files.models import *
from yolo_object_detector_test_files.utils.utils import *
from yolo_object_detector_test_files.utils.datasets import *
from yolo_object_detector_test_files.utils.augmentations import *
from yolo_object_detector_test_files.utils.transforms import *
from yolo_object_detector_test_files.sort import *

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
config_path='yolo_object_detector/yolo_object_detector_test_files/yolov3-custom.cfg'
weights_path='yolo_object_detector/yolo_object_detector_test_files/YOLO_V3_PORPOISE_MODEL.pth'
class_path='yolo_object_detector/yolo_object_detector_test_files/classes.names'
#video_path='videos/20200417 - Male - Group of porpoises bottom feeding.MOV'
#video_path='videos/20190629_Kerteminde_bay_Three_porpoises_Dennis_data.MOV'
#video_path='videos/20190407_Male_DJI4_Three_porpoises.MOV'
video_path="videos/20200417 - Male - Group of porpoises with calf foraging.MOV"
START_FRAME = 79
img_size=608
conf_thres=0.8
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
capture.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

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


if not capture.isOpened():
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
    cv2.imwrite("yolo_test_79_calf_for.jpg", frame)
    cv2.waitKey(0)
        

