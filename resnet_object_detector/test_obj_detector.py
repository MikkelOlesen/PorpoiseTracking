import torchvision
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import time


from PIL import Image
from libs.sort import *

video_path='videos/20190319_Male_Group_of_porpoises_and_a_calf.MOV'
START_FRAME = 0
CONF = 0.5

transform = transforms.Compose([
    #transforms.Resize(800),
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image

    #boz = torchvision.ops.nms(outputs[0]['boxes'],outputs[0]['scores'], 0.3)
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold]
    scores = pred_scores[pred_scores >= detection_threshold]
    scores = np.round(scores, decimals=3)

    return boxes, scores

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load("resnet_object_detector/models/model_50_eproc")
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
        boxes, score = predict(pilimg, model, device, CONF)
        

        if boxes is not None:
            #combine boxes and scores for MO tracker to use
            bBoxes = np.insert(boxes, 4, score, axis=1)

            track_bBoxes_id = mot_tracker.update(bBoxes)

            for x1, y1, x2, y2, score in bBoxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,128),2)
                cv2.putText(frame,str(score),(int(x2), int(y2)),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),thickness=3)
            
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()