import torchvision
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import time
from pathlib import Path


from PIL import Image
from libs.sort import *


START_FRAME = 0
CONF = 0.5

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Porpoise extractor - Extract porpoises from video files and save to img')
    parser.add_argument('input_path', help='Path to the input file (video)', type=str)
    parser.add_argument("output_path", help="Path to output folder", type=str)
    parser.add_argument("model", help="Path to model", type=str)
    parser.add_argument("frame_spaceing", help="Set program to extract every n frame (default = 1)", type=int)
    args = parser.parse_args()
    return args

transform = transforms.Compose([
    #transforms.Resize(800),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
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

def main():
    args = parse_args()
    video_path = args.input_path
    video_name = Path(args.input_path)
    output_path = args.output_path
    model_path = args.model
    frame_spaceing = args.frame_spaceing
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(model_path)
    model.eval().to(device)

    #Load video
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    if not capture.isOpened():
        print("Unable to open video")
        exit(0)
    while True:
        ret, frame = capture.read()
        frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        
        if frame is None:
            break

        if(not(frame_number % frame_spaceing)):
            #Convert image to pil image and detect porpoises
            pilimg = Image.fromarray(frame)
            boxes, score = predict(pilimg, model, device, CONF)
            
            if boxes is not None:
                #combine boxes and scores
                bBoxes = np.insert(boxes, 4, score, axis=1)

                for count, (x1, y1, x2, y2, score) in enumerate(bBoxes):
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    cv2.imwrite(output_path + video_name.stem + "_" + str(frame_number) +"_"+ str(count) + ".jpg",crop)

    capture.release()

if __name__ == "__main__":
    main()