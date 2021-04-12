from __future__ import print_function
from time import sleep
import os
dirname = os.path.dirname(__file__)
import cv2
import numpy as np
import matplotlib as plt
import imutils
from sort import *

def calc_sd(vals, mean):
    valSum = 0
    for val in vals:
        valSum += ((val - mean) ** 2)
    sd = np.sqrt(1/(len(vals) - 1) * valSum)
    return sd

def splitChannels(colPixs, img):
    b_vals = []
    g_vals = []
    r_vals = []

    for pt in colPixs:
        b_vals.append(img[pt[0][1], pt[0][0]][0])
        g_vals.append(img[pt[0][1], pt[0][0]][1])
        r_vals.append(img[pt[0][1], pt[0][0]][2])

    return b_vals, g_vals, r_vals

def save_crop(img,cnts,frame_number):
    i = 0
    for cnt in cnts:
        i += 1
        M = cv2.moments(cnt)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
        
        _, radius = cv2.minEnclosingCircle(cnt)
        size = int(radius * 2.5)

        if(y-size < 0):
            continue
        elif(y+size > img.shape[0]):
            continue
        elif(x-size < 0):
            continue
        elif(x+size > img.shape[1]):
            continue
        crop_img = img[y-size:y+size, x-size:x+size]
        cv2.imwrite("output/" + str(int(frame_number)) + "_" + str(i) + ".jpg",crop_img)

def create_bb_array(frame,cnts):
    bBoxes_list = []
    for cnt in cnts:
        bBox = cv2.boundingRect(cnt)
        bBox = [bBox[0], bBox[1], bBox[0]+bBox[2], bBox[1]+bBox[3], 1] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        bBoxes_list.append(bBox)
    bBoxes = np.array(bBoxes_list)
    return bBoxes

def track_cnt(cnts):
    pass


mot_tracker = Sort(6,3,0.3)
filename = os.path.join(dirname, '../Videos/20190319_Male_Group_of_porpoises_and_a_calf.MOV')
capture = cv2.VideoCapture(filename)
maskfilename = os.path.join(dirname, 'water_mask.PNG')
reffilename = os.path.join(dirname, 'ref_img.PNG')
imgMask = cv2.imread(maskfilename)
imgRef = cv2.imread(reffilename)

#Lab color space
imgRef = cv2.cvtColor(imgRef, cv2.COLOR_BGR2LAB)

# Select pixels for mask (Entire img)
#print("Extracting pixels for mask")
#imgMask = cv2.inRange(imgMask, (0,0,220),(40,40,255))

#spilt the channels
#colPixs = cv2.findNonZero(imgMask)
#b_vals, g_vals, r_vals = splitChannels(colPixs, imgRef)
'''
#Calculate mean for each channel
print("Calculating mean")
b_mean = np.mean(b_vals)
g_mean = np.mean(g_vals)
r_mean = np.mean(r_vals)

#Calculate standard deviation for each channel
print("Calculating std diviation")
b_sd = calc_sd(b_vals, b_mean)
g_sd = calc_sd(g_vals, g_mean)
r_sd = calc_sd(r_vals, r_mean)
'''
b_mean = 136.00291959813308
g_mean = 106.83865168143082
r_mean = 135.36836851093034
b_sd = 13.575024881816251
g_sd = 3.8573182016679635
r_sd = 3.9693297640228216

#print(b_mean, g_mean, r_mean, b_sd, g_sd, r_sd)

''' Number of standard deviations to include '''
std_div_fact = 5

if not capture.isOpened:
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
    bgMask = cv2.inRange(calcframe, (b_mean - b_sd * std_div_fact, g_mean - g_sd * std_div_fact, r_mean - r_sd * std_div_fact), (b_mean + b_sd * std_div_fact, g_mean + g_sd * std_div_fact, r_mean + r_sd * std_div_fact))
    fgmask = cv2.bitwise_not(bgMask)


    #cv2.namedWindow("masked img",cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("masked img",fgmask)

    kernel = np.ones((7,7), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    

    items = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(items)
    #if not cnts:
    #    print("No animals")
    #    continue

    cnt = max(cnts, key = cv2.contourArea)
    max_contourArea = cv2.contourArea(cnt)
    #if max_contourArea < 300:
    #    print("Not animal too small")
    #    continue

    temp_cnts = []
    
    for c in cnts:
        contourArea = cv2.contourArea(c)
        if contourArea > 0.25 * max_contourArea and contourArea > 300:
            temp_cnts.append(c)

    cnts = temp_cnts

    #if len(cnts) > 6:
    #    print("Not animals")
    #    continue

    #save_crop(frame,cnts,frame_number)
    #cv2.drawContours(frame, cnts, -1, (0,0,255), 3)
    bBoxes = create_bb_array(frame,cnts)
    #sleep()

    
    
    track_bBoxes_id = mot_tracker.update(bBoxes)
    #print(frame_number)
    #print(track_bBoxes_id)


    for bb in track_bBoxes_id:
        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]),int(bb[3])), (255,0,0),3)
        cv2.putText(frame,str(int(bb[4])),(int(bb[2]),int(bb[3])),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0),thickness=6)
 




    #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.rectangle(frame, (10, 2), (300,100), (255,255,255), -1)
    cv2.putText(frame, str(int(capture.get(cv2.CAP_PROP_POS_FRAMES))), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),thickness=6)
    #cv2.putText(frame, str(int(fps)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),thickness=6)

    
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