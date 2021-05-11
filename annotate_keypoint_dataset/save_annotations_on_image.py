import cv2
import os
import argparse

Pts = []

def load_keypoint_annotations(path):
    """
        Loads keypoints in annotation file at 'path'
        Returns keypoints
    """
    keypoints = []
    with open(path, 'r') as file:
        for row in file:
            x, y , _ = row.split()
            keypoints.append([float(x), float(y)])

    return keypoints


def load_images_from_folder(folder):
    '''
    Returns a list of image filenames in folder
    '''
    imgs = list(sorted(os.listdir(folder)))
    return imgs

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Keypoint annotator - annotate Head, left fin and right fin on images of porpoises. \n\n   Image folder must only contain image files. \n\n   Keypoints are added to file in selected order. Therefore select in order: Head, leftfin, rightfin.', epilog="OBS: input and output folder can not be the same, would result in error if runned multiple times.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('image_folder', help='Path to input folder of images', type=str)
    parser.add_argument("anno_folder", help="Path to output folder for the annotation files", type=str)
    parser.add_argument("output_folder", help="Path to output folder for the annotation files", type=str)
    args = parser.parse_args()
    return args

def annotate_img(event, x, y, flags, param):
    global Pts
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(Pts) < 8:
            radius = 8
            cv2.circle(param,(x,y),radius,(0,255,0), 1)
            cv2.line(param,(x-radius,y),(x+radius,y),(0,255,0),1)
            cv2.line(param,(x,y-radius),(x,y+radius),(0,255,0),1)
            Pts.append(x)
            Pts.append(y)
        

def main():
    #Load arguments
    args = parse_args()
    input_path = args.image_folder
    anno_path = args.anno_folder
    output_path = args.output_folder

    #Load a list of the images to show
    image_list = load_images_from_folder(input_path)

    
    for img_name in image_list:
        img_path = str(os.path.join(os.path.abspath(os.getcwd()),input_path))
        anno_path = str(os.path.join(os.path.abspath(os.getcwd()),anno_path))
        out_path = str(os.path.join(os.path.abspath(os.getcwd()),output_path))

        img = cv2.imread(img_path +"/"+ img_name)        

        #Get annotation file for the image
        annotation_name = os.path.splitext(img_name)[0] + ".txt"
        annotation_name = os.path.join(anno_path,annotation_name)
        keypoints = load_keypoint_annotations(annotation_name)
        
        #Draw keypoints on image
        for i, (x, y) in enumerate(keypoints):
                t = i + 1
                cv2.circle(img, (int(x),int(y)), radius=2, color=(int(255/t),0,int((255/4) * t)), thickness=-1)

        #save image
        cv2.imwrite(out_path + img_name, img)



if __name__ == "__main__":
    main()