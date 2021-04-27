# PorpoiseTracking
Tracking of porpoises

## Content
All of the trackers and object detetors uses [Sort](https://github.com/abewley/sort) to track the detetions/porpoises over time. The sort libary is not created by me. 

* ### /annotate_keypoint_dataset
    annotate_keypoint_dataset contains the files for annotating image files of porpoises with keypoints and save them as a .txt file.

* ### /simple_tracker
    Contains the first tracker implemented. The simple tracker uses segmentation by color thresholding. The porpoises are segmented from the background by the mean color of the ocean.

* ### /export_frames_from_video
    export_frames_from_video contains files for exporting individual videoframes as images.

* ### /yolo_object_detector
    yolo_object_detector contains the files for the yolo object detector. This object detector uses [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) library as a backbone. 

* ### /object_detection
    object_detection contains files for the final object detection implementation with a Faster R-CNN model with a ResNet-50-FPN backbone. The folder contains both a file for training the a model and a file for testing it on a video file.

* ### /keypoint_detection
    keypoint_detection contains the files for traning a model that can detect keypoints on a porpoise.
