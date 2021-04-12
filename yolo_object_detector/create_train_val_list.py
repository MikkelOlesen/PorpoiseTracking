import glob
import os
import numpy as np
import sys
import random


drive_dir = "/content/PyTorch-YOLOv3/data/custom/images"
img_dir = "Frame_export/comb_imgs"

split_pct = 10  # 10% validation set

file_train = open("train.txt", "w")  
file_val = open("valid.txt", "w")  

counter = 1  
index_test = round(100 / split_pct)
for fullpath in glob.iglob(os.path.join(img_dir, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(fullpath))
    if counter == index_test:
        counter = 1
        file_val.write(drive_dir + "/" + title + '.jpg' + "\n")
    else:
        file_train.write(drive_dir + "/" + title + '.jpg' + "\n")
        counter = counter + 1

file_train.close()
file_val.close()
file_train = open("train.txt", "r")  
file_val = open("valid.txt", "r")  

lines_val = file_val.readlines()
random.shuffle(lines_val)

lines_train = file_train.readlines()
random.shuffle(lines_train)

file_train.close()
file_val.close()
file_train = open("train.txt", "w")  
file_val = open("valid.txt", "w")  

file_val.writelines(lines_val)
file_train.writelines(lines_train)
file_train.close()
file_val.close()


