import torchvision
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import time
import torchvision.transforms.functional as F
from pathlib import Path
import argparse


from PIL import Image
from libs.sort import *


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Porpoise tracker with keypoints. Takes a video and annotates bounding boxes, as well as head, left tailfin, right tailfin, and center tailfin of porpoises. \n\n  The output of the program is the video and a CSV file with the annotations. \n The output will be placed in the same folder as the input video.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('video_path', help='Path to video', type=str)
    parser.add_argument('detection_model_path', help='Path to porpoise detection model', type=str)
    parser.add_argument('keypoint_model_path', help='Path to keypoint regression model ', type=str)
    parser.add_argument("confidence_threshold", help="Confidence threshold for detecting porpoises", type=float)
    parser.add_argument("show_output", help="Show output in a window while the program is running. True or False", type=str)
    args = parser.parse_args()
    return args

porpoise_detection_transform = transforms.Compose([
    transforms.Resize((800,800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        image = F.pad(image, padding, 0, 'constant')

        return image

keypoint_detection_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def transform_keypoints_to_original_size(orig_image, keypoints):
    [[x1, y1, x2, y2, x3, y3, x4, y4]] = keypoints 
    orig_image = Image.fromarray(orig_image)

    #get padding
    w, h = orig_image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)

    #Get rescale value
    rescale_value = max_wh/224

    #transform keypoints
    keypoints = np.array([(x1*rescale_value)-hp, (y1*rescale_value)-vp, (x2*rescale_value)-hp, (y2*rescale_value)-vp, (x3*rescale_value)-hp, (y3*rescale_value)-vp, (x4*rescale_value)-hp, (y4*rescale_value)-vp])

    return keypoints.astype(int)

def find_porpoises(image, model, device, detection_threshold):
    # transform the image to tensor
    image = porpoise_detection_transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension

    with torch.no_grad():
        outputs = model(image) # get the predictions on the image
    
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold]
    scores = pred_scores[pred_scores >= detection_threshold]
    scores = np.round(scores, decimals=3)

    return boxes, scores

def find_keypoints(image, model, device):
    #transform from opencv to pil
    image = Image.fromarray(image)
    
    #transform image to be 224x224 with padding
    image = keypoint_detection_transform(image).to(device)

    #Add batch dimention (one batch)
    image = image.unsqueeze(0)

    with torch.no_grad():
        keypoints = model(image).cpu().detach().numpy()

    return keypoints

def draw_skeleton(image, keypoints, offsetX, offsetY):
    [x1, y1, x2, y2, x3, y3, x4, y4] = keypoints

    #Draw points
    cv2.circle(image, ((x1 + offsetX),(y1 + offsetY)), radius=3, color=(0,0,255), thickness=-1)
    cv2.circle(image, ((x2 + offsetX),(y2 + offsetY)), radius=3, color=(0,0,255), thickness=-1)
    cv2.circle(image, ((x3 + offsetX),(y3 + offsetY)), radius=3, color=(0,0,255), thickness=-1)
    cv2.circle(image, ((x4 + offsetX),(y4 + offsetY)), radius=3, color=(0,0,255), thickness=-1)

    #Draw skeleton
    cv2.line(image, ((x1 + offsetX),(y1 + offsetY)), ((x2 + offsetX),(y2 + offsetY)), (255,0,0), 2)
    cv2.line(image, ((x2 + offsetX),(y2 + offsetY)), ((x3 + offsetX),(y3 + offsetY)), (255,0,0), 2)
    cv2.line(image, ((x2 + offsetX),(y2 + offsetY)), ((x4 + offsetX),(y4 + offsetY)), (255,0,0), 2)

    return image

def transform_boxes_to_original_size(orig_image, boxes):
    h, w, c = orig_image.shape
    new_bboxes = []

    for x1, y1, x2, y2 in boxes:
        x1 = x1 * (w/800)
        y1 = y1 * (h/800)
        x2 = x2 * (w/800)
        y2 = y2 * (h/800)
        new_bboxes.append([x1, y1, x2, y2])

    return new_bboxes


def main():
    args = parse_args()
    video_path = args.video_path
    CONF = args.confidence_threshold
    obj_detection_model = args.detection_model_path
    key_regression_model = args.keypoint_model_path
    show_output = args.show_output

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #Load models
    detection_model = torch.load(obj_detection_model)
    detection_model.eval().to(device)

    keypoint_model = torch.load(key_regression_model)
    keypoint_model.eval().to(device)

    #Load video
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    #Create video writer
    video_path_p = Path(video_path)
    output_video = cv2.VideoWriter(str(video_path_p.parent) + "/" + str(video_path_p.stem) + '_annotated.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    #create MO tracker
    mot_tracker = Sort(6,3,0.3) # max age out of frame, min amount of hits to start tracking, Minimum IOU for match

    if not capture.isOpened():
        print("Unable to open video")
        exit(0)
    while True:
        ret, frame = capture.read()
        frame_number = capture.get(cv2.CAP_PROP_POS_FRAMES)
        
        if frame is None:
            break

        frame_copy = frame.copy()
        
        #Convert image to pil image and detect porpoises
        pilimg = Image.fromarray(frame)
        boxes, score = find_porpoises(pilimg, detection_model, device, CONF)
        #Rescale boxes to fit original video
        boxes = transform_boxes_to_original_size(frame, boxes)
        
        if len(boxes) > 0:
            #combine boxes and scores for MO tracker to use
            bBoxes = np.insert(boxes, 4, score, axis=1)

            track_bBoxes_id = mot_tracker.update(bBoxes)

            #Draw predicted box
            for x1, y1, x2, y2, score in bBoxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,128),2)
                cv2.putText(frame,str(round(score,2)),(int(x2), int(y2)),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),thickness=3)
            
            #Draw tracked box and find keypoints
            for bb in track_bBoxes_id:
                #draw bbox
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]),int(bb[3])), (0,255,0),2)
                cv2.putText(frame, str(int(bb[4])), (int(bb[0]),int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),thickness=3)
                
                #make Bbox a new image
                bbox_image = frame_copy[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]

                if (0 not in bbox_image.shape): #if the bbox is not out of the image frame
                    #find keypoints
                    keypoints = find_keypoints(bbox_image, keypoint_model, device)
                    #transform keypoints 
                    keypoints = transform_keypoints_to_original_size(bbox_image, keypoints)
                    # draw skeleton
                    frame = draw_skeleton(frame, keypoints, int(bb[0]), int(bb[1]))
                    #Write data to csv file.
                    [x1, y1, x2, y2, x3, y3, x4, y4] = keypoints 
                    with open(str(video_path_p.parent) + "/" + str(video_path_p.stem) + "_annotations.csv", "a") as file:
                        file.write(str(int(frame_number)) + "," + str(int(bb[4])) + "," + str(int(bb[0])) + "," + str(int(bb[1])) + "," + str(int(bb[2])) + "," + str(int(bb[3])) + "," + str(int(x1)) + "," + str(int(y1)) + "," + str(int(x2)) + "," + str(int(y2)) + "," + str(int(x3)) + "," + str(int(y3)) + "," + str(int(x4)) + "," + str(int(y4)) + "\n")

        #cv2.rectangle(frame, (10, 2), (300,100), (255,255,255), -1)
        #cv2.putText(frame, str(int(capture.get(cv2.CAP_PROP_POS_FRAMES))), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),thickness=6)

        #Rescale and display video
        if(int(show_output)):
            x = 3
            cv2.namedWindow("Frame",cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('Frame', int(frame.shape[1]/x),int(frame.shape[0]/x))
            cv2.imshow("Frame", frame)

        #Save to video file
        output_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    capture.release()
    output_video.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()