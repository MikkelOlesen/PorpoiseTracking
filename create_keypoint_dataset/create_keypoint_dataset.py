import cv2
import os
import argparse

Pts = []

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
    parser.add_argument("output_folder", help="Path to output folder for the annotation files", type=str)
    args = parser.parse_args()
    return args

def annotate_img(event, x, y, flags, param):
    global Pts
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(Pts) < 6:
            radius = 10
            cv2.circle(param,(x,y),radius,(0,255,0), 1)
            cv2.line(param,(x-radius,y),(x+radius,y),(0,255,0),1)
            cv2.line(param,(x,y-radius),(x,y+radius),(0,255,0),1)
            Pts.append(x)
            Pts.append(y)
        

def main():
    args = parse_args()
    input_path = args.image_folder
    output_path = args.output_folder

    image_list = load_images_from_folder(input_path)
    end = False
    global Pts

    for img_name in image_list:
        img_path = str(os.path.join(os.path.abspath(os.getcwd()),input_path))
        out_path = str(os.path.join(os.path.abspath(os.getcwd()),output_path))
        img = cv2.imread(img_path + img_name)
        clone = img.copy()
        anno_name = os.path.splitext(img_name)[0] + ".txt"

        cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(img_name, 512, 512)

        while True:
            
            cv2.imshow(img_name, img)
            cv2.setMouseCallback(img_name, annotate_img, img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                end = True
                break
            elif key == ord('d') and (len(Pts) > 5):
                with open(out_path + anno_name, 'w') as file:
                    file.write(str(Pts[0]) + " " + str(Pts[1]) + " " + str(Pts[2]) + " " + str(Pts[3]) + " " + str(Pts[4]) + " " + str(Pts[5]))
                Pts = []
                cv2.destroyAllWindows()
                break
            elif key == ord('r'):
                Pts = []
                img = clone.copy()
                cv2.destroyAllWindows()
                cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(img_name, 512, 512)

        if end:
            break   


if __name__ == "__main__":
    main()