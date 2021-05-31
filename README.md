# PorpoiseTracking
This reposetory contains the files for Mikkel Dupont Olesen's master thesis at Southern University of Denmark. 

## Dependencies
To install required dependencies run:
```
pip install -r requirements.txt
```
The program also uses Pytorch to install pytorch follow the instructions on pytorch [homepage](https://pytorch.org/get-started/locally/).
The run the detection and training on a GPU install CUDA toolkit, which can be downloaded [here](https://developer.nvidia.com/cuda-toolkit).

## Run porpoise tracker and pose estimator
To run the porpoise tracker and pose estimator run the script ```porpoise_tracker_with_keypoints.py``` placed in porpoise_tracker_with_keypoints. The script can be runned with the following command. 
```
python porpoise_tracker_with_keypoints.py <video_path> <detection_model_path> <keypoint_model_path> <confidence_threshold> <show_output>
```



## Content
All of the trackers and object detetors uses [Sort](https://github.com/abewley/sort) to track the detetions/porpoises over time. The sort libary is not created by me. 

* ### /annotate_keypoint_dataset
    annotate_keypoint_dataset contains the files for annotating image files of porpoises with keypoints and save them as a .txt file.

* ### /simple_tracker
    Contains the first tracker implemented. The simple tracker uses segmentation by color thresholding. The porpoises are segmented from the background by the mean color of the ocean.

* ### /export_frames_from_video
    export_frames_from_video contains files for exporting individual videoframes as images.

* ### /yolo_object_detector
    yolo_object_detector contains the files for the yolo object detector. This object detector uses [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) library to train the object detector.

* ### /object_detection
    object_detection contains files for the final object detection implementation with a Faster R-CNN model with a ResNet-50-FPN backbone. The folder contains both a file for training the a model and a file for testing it on a video file.

* ### /keypoint_detection
    keypoint_detection contains the files for traning a model that can detect keypoints on a porpoise, and a files for testing the model on single images. 
    
* ### /porpoise_tracker_with_keypoints
    Contains the files for running the porpoise tracker and pose estimator. 
