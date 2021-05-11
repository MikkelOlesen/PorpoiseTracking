import torchvision
import numpy as np
import torch
import cv2
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        image = F.pad(image, padding, 0, 'constant')

        return image

transform = transforms.Compose([
    SquarePad(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


def transform_keypoints_to_original_size(orig_image, keypoints):
    [[x1, y1, x2, y2, x3, y3, x4, y4]] = keypoints 

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


def main():
    #Load model and run on GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load("keypoint_detection/model_R34_5_9")
    model.eval().to(device)

    image = Image.open("keypoint_detection/test.JPG")
    model_image = transform(image).to(device)

    #Add batch dimention (one batch)
    model_image = model_image.unsqueeze(0)

    #Run model on image
    with torch.no_grad():
        keypoints = model(model_image).cpu().detach().numpy()

    keypoints = transform_keypoints_to_original_size(image, keypoints)
    [x1, y1, x2, y2, x3, y3, x4, y4] = keypoints
    
    #Convert to opencv Image
    cv_image = np.array(image)
    cv_image = cv_image[:, :, ::-1].copy()

    cv2.circle(cv_image, (x1,y1), radius=2, color=(0,0,255), thickness=-1)
    cv2.circle(cv_image, (x2,y2), radius=2, color=(0,0,255), thickness=-1)
    cv2.circle(cv_image, (x3,y3), radius=2, color=(0,0,255), thickness=-1)
    cv2.circle(cv_image, (x4,y4), radius=2, color=(0,0,255), thickness=-1)


    cv2.imshow("Image", cv_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()