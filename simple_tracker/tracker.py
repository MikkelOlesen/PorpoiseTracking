from __future__ import print_function
from time import sleep
import os
dirname = os.path.dirname(__file__)
import cv2
import numpy as np
import matplotlib as plt
import imutils
from Libs.sort import *

def calc_sd(vals, mean): #calculate standard diviation
    valSum = 0
    for val in vals:
        valSum += ((val - mean) ** 2)
    sd = np.sqrt(1/(len(vals) - 1) * valSum)
    return sd

def splitChannels(colPixs, img): #Spilt the image into 3 color channels
    l_vals = []
    a_vals = []
    b_vals = []

    for pt in colPixs:
        l_vals.append(img[pt[0][1], pt[0][0]][0])
        a_vals.append(img[pt[0][1], pt[0][0]][1])
        b_vals.append(img[pt[0][1], pt[0][0]][2])

    return l_vals, a_vals, b_vals

def create_bb_array(frame,cnts):
    bBoxes_list = []
    for cnt in cnts:
        bBox = cv2.boundingRect(cnt)
        bBox = [bBox[0], bBox[1], bBox[0]+bBox[2], bBox[1]+bBox[3], 1] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        bBoxes_list.append(bBox)
    bBoxes = np.array(bBoxes_list)
    return bBoxes


#Create Multi object tracker
mot_tracker = Sort(6,3,0.3)

#load video
filename = os.path.join(dirname, '../videos/training_data/20190319_Male_Group_of_porpoises_and_a_calf.MOV')
capture = cv2.VideoCapture(filename)
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

#Load images for masking ocean
maskfilename = os.path.join(dirname, 'water_mask.PNG')
reffilename = os.path.join(dirname, 'ref_img.PNG')
imgMask = cv2.imread(maskfilename)
imgRef = cv2.imread(reffilename)

#Convert to Lab color space
imgRef = cv2.cvtColor(imgRef, cv2.COLOR_BGR2LAB)

# Select pixels for mask (Entire img)
print("Extracting pixels for mask")
imgMask = cv2.inRange(imgMask, (0,0,220),(40,40,255))

#spilt the channels
colPixs = cv2.findNonZero(imgMask)
l_vals, a_vals, b_vals = splitChannels(colPixs, imgRef)

#Calculate mean for each channel
print("Calculating mean")
l_mean = np.mean(l_vals)
a_mean = np.mean(a_vals)
b_mean = np.mean(b_vals)

#Calculate standard deviation for each channel
print("Calculating std diviation")
l_sd = calc_sd(l_vals, l_mean)
a_sd = calc_sd(a_vals, a_mean)
b_sd = calc_sd(b_vals, b_mean)

# USED TO SKIP COLOR CALCULATION
#l_mean = 136.00291959813308
#a_mean = 106.83865168143082
#b_mean = 135.36836851093034
#l_sd = 13.575024881816251
#a_sd = 3.8573182016679635
#b_sd = 3.9693297640228216
#print(l_mean, a_mean, b_mean, l_sd, a_sd, b_sd)

''' Number of standard deviations to include '''
std_div_fact = 5

if not capture.isOpened():
    print("Unable to open video")
    exit(0)
while True:
    #timer = cv2.getTickCount()
    ret, frame = capture.read()

    frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)

    if frame is None:
        break

    calcframe = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    #Mask
    bgMask = cv2.inRange(calcframe, (l_mean - l_sd * std_div_fact, a_mean - a_sd * std_div_fact, b_mean - b_sd * std_div_fact), (l_mean + l_sd * std_div_fact, a_mean + a_sd * std_div_fact, b_mean + b_sd * std_div_fact))
    fgmask = cv2.bitwise_not(bgMask)

    #Remove noise
    kernel = np.ones((7,7), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    #Find Contours
    items = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(items)

    #Find biggist contour 
    cnt = max(cnts, key = cv2.contourArea)
    max_contourArea = cv2.contourArea(cnt)

    #Remove contour if smaller than 25% of biggist porpoise
    temp_cnts = []
    for c in cnts:
        contourArea = cv2.contourArea(c)
        if contourArea > 0.25 * max_contourArea and contourArea > 300:
            temp_cnts.append(c)
    cnts = temp_cnts

    #Create Bounding boxes from contours
    bBoxes = create_bb_array(frame,cnts)

    
    #Track bounding boxes
    track_bBoxes_id = mot_tracker.update(bBoxes)

    #Draw bounding boxes
    for bb in track_bBoxes_id:
        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]),int(bb[3])), (255,0,0),3)
        cv2.putText(frame,str(int(bb[4])),(int(bb[2]),int(bb[3])),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0),thickness=6)
 
    # Frame counter
    cv2.rectangle(frame, (10, 2), (300,100), (255,255,255), -1)
    cv2.putText(frame, str(int(capture.get(cv2.CAP_PROP_POS_FRAMES))), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),thickness=6)

    x = 4
    cv2.namedWindow("Frame",cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Mask",cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Frame', int(frame.shape[1]/x),int(frame.shape[0]/x))
    cv2.resizeWindow('Mask', int(fgmask.shape[1]/x),int(fgmask.shape[0]/x))
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", fgmask)
        
    keyboard = cv2.waitKey(5)
    if keyboard == 'q' or keyboard == 27:
        break