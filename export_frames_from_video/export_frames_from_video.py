import argparse
import cv2
from time import sleep

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Frame extractor - Extract frames from video files every n frames')
    parser.add_argument('input_path', help='Path to the input file (video)', type=str)
    parser.add_argument("output_path", help="Path to output folder", type=str)
    parser.add_argument("frame_spaceing", help="Set program to extract every n frame (default = 1)", type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    frame_spaceing = args.frame_spaceing

    capture = cv2.VideoCapture(input_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


    if not capture.isOpened():
        print("Unable to open video")
        exit(0)
    
    for i in range(total_frames):
        ret, frame = capture.read()

        frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        
        if frame is None:
            break

        if(not(frame_number % frame_spaceing)):
            cv2.imwrite(output_path + str(frame_number) + ".jpg",frame)
            print("Exported frame: " + str(frame_number) + " / " + str(total_frames) + " -- " + output_path + str(frame_number) + ".jpg")
        
