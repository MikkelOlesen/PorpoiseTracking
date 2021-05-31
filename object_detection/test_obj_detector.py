import torchvision
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import time


from PIL import Image
from Libs.sort import *

video_path='videos/20200417 - Male - Group of porpoises with calf foraging.MOV'
START_FRAME = 79
CONF = 0.8

transform = transforms.Compose([
    transforms.Resize((800,800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_porpoises(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load("object_detection/models/model_643_normal_anchor")
    model.eval().to(device)

    #Load video
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    
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
        
        #Convert image to pil image and detect porpoises
        pilimg = Image.fromarray(frame)
        boxes, score = detect_porpoises(pilimg, model, device, CONF)
        boxes = transform_boxes_to_original_size(frame,boxes)
        
        
        if len(boxes) > 0:
            #combine boxes and scores for MO tracker to use
            bBoxes = np.insert(boxes, 4, score, axis=1)

            track_bBoxes_id = mot_tracker.update(bBoxes)

            #Draw predicted box
            for x1, y1, x2, y2, score in bBoxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,128),2)
                cv2.putText(frame,str(round(score,3)),(int(x2), int(y2)),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),thickness=3)
            
            #Draw tracked box
            for bb in track_bBoxes_id:
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]),int(bb[3])), (0,255,0),2)
                cv2.putText(frame,str(int(bb[4])),(int(bb[0]),int(bb[1])),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),thickness=3)
        

        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.rectangle(frame, (10, 2), (300,100), (255,255,255), -1)
        cv2.putText(frame, str(int(capture.get(cv2.CAP_PROP_POS_FRAMES))), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),thickness=6)
        #cv2.putText(frame, str(int(fps)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),thickness=6)

        x = 3
        cv2.namedWindow("Frame",cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Frame', int(frame.shape[1]/x),int(frame.shape[0]/x))
        cv2.imshow("Frame", frame)
        cv2.imwrite("faster_rcnn_test_3.png", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()