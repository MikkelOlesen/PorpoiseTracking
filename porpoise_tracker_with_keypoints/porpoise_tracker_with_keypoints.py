import torchvision
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import time
import torchvision.transforms.functional as F


from PIL import Image
from libs.sort import *

video_path="videos/20200417 - Male - Group of porpoises bottom feeding.MOV"
START_FRAME = 1325
CONF = 0.5

porpoise_detection_transform = transforms.Compose([
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


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #Load models
    detection_model = torch.load("porpoise_tracker_with_keypoints/models/model_30_eproc_resnet")
    detection_model.eval().to(device)

    keypoint_model = torch.load("porpoise_tracker_with_keypoints/models/model_R34_5_9")
    keypoint_model.eval().to(device)

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
        frame_copy = frame.copy()

        if frame is None:
            break
        
        
        #Convert image to pil image and detect porpoises
        pilimg = Image.fromarray(frame)
        boxes, score = find_porpoises(pilimg, detection_model, device, CONF)
        
        
        if boxes is not None:
            #combine boxes and scores for MO tracker to use
            bBoxes = np.insert(boxes, 4, score, axis=1)

            track_bBoxes_id = mot_tracker.update(bBoxes)

            #Draw predicted box
            for x1, y1, x2, y2, score in bBoxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,128),2)
                cv2.putText(frame,str(score),(int(x2), int(y2)),cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0),thickness=3)
            
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

                    frame = draw_skeleton(frame, keypoints, int(bb[0]), int(bb[1]))

        
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