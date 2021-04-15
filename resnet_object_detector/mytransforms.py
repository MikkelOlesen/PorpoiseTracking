import random
import numpy as np
from torchvision.transforms import functional as F
import cv2
import torch
import math
import time

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = int(size)
    def __call__(self, image, target):
        _, h, w = image.size()
        image = F.resize(image, (self.size, self.size))
        bbox = target['boxes']
        num_bboxes = len(bbox)

        for i in range(num_bboxes):
            bbox.data[i][0] = bbox.data[i][0] * self.size/w
            bbox.data[i][1] = bbox.data[i][1] * self.size/h
            bbox.data[i][2] = bbox.data[i][2] * self.size/w
            bbox.data[i][3] = bbox.data[i][3] * self.size/h

        target['boxes'] = bbox
        return image, target

class RandomHorizontalFlip(object):
    
    def __init__(self, probability):
        self.prob = probability

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = F.hflip(image)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox    
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, probability):
        self.prob = probability

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = F.vflip(image)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
        return image, target

class RandomColor(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, image, target):
        ran1 = random.uniform(1.0-self.brightness, 1.0+self.brightness)
        ran2 = random.uniform(1.0-self.contrast, 1.0+self.contrast)
        ran3 = random.uniform(1.0-self.saturation, 1.0+self.saturation)
        ran4 = random.uniform(-self.hue, self.hue)

        image = F.adjust_brightness(image, ran1)
        image = F.adjust_contrast(image, ran2)
        image = F.adjust_saturation(image, ran3)
        image = F.adjust_hue(image, ran4)
        return image, target

class ShowImg(object):
    def __call__(self, image, target):
        mean = np.array([0.485, 0.456, 0.406])
        std =  np.array([0.229, 0.224, 0.225])

        boxes = target['boxes'].detach().cpu().numpy()
        pil_image = F.to_pil_image(image)

        open_cv_image = np.array(pil_image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        for x1, y1, x2, y2 in boxes:
                cv2.rectangle(open_cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,128),2)

        cv2.imshow("Augmentented", open_cv_image)
        cv2.waitKey(0)
        
        return None

    