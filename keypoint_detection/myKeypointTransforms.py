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

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        for t in self.transforms:
            sample = t(sample)
        
        image, keypoints = sample['image'], sample['keypoints']
        return {'image': image, 'keypoints': keypoints}

class ToTensor(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = F.to_tensor(image)
        return {'image': image, 'keypoints': keypoints}

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'keypoints': keypoints}

class Resize(object):
    def __init__(self, size):
        self.size = int(size)
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        _, h, w = image.size()
        image = F.resize(image, (self.size, self.size))

        for i in range(len(keypoints)):
            keypoints.data[i][0] = keypoints.data[i][0] * self.size/w
            keypoints.data[i][1] = keypoints.data[i][1] * self.size/h

        return {'image': image, 'keypoints': keypoints}

class Square_Pad:
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # pad image
        _, h, w = image.size()
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp) # (padding_left, padding_top, padding_right, padding_bottom)
        image = F.pad(image, padding, 0, 'constant')

        #adjust keypoints depending on padding
        for i in range(len(keypoints)):
            keypoints.data[i][0] = keypoints.data[i][0] + hp
            keypoints.data[i][1] = keypoints.data[i][1] + vp


        return {'image': image, 'keypoints': keypoints}


class RandomColor(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        ran1 = random.uniform(1.0-self.brightness, 1.0+self.brightness)
        ran2 = random.uniform(1.0-self.contrast, 1.0+self.contrast)
        ran3 = random.uniform(1.0-self.saturation, 1.0+self.saturation)
        ran4 = random.uniform(-self.hue, self.hue)

        image = F.adjust_brightness(image, ran1)
        image = F.adjust_contrast(image, ran2)
        image = F.adjust_saturation(image, ran3)
        image = F.adjust_hue(image, ran4)
        return {'image': image, 'keypoints': keypoints}

class AddRandomNoise(object):
    def __init__(self, factor, prob):
        self.factor = factor
        self.prob = prob

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        if random.random() < self.prob:
            noise = torch.randn_like(image)
            image = image + noise * self.factor
        return {'image': image, 'keypoints': keypoints}


class ShowImg(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        mean = np.array([0.485, 0.456, 0.406])
        std =  np.array([0.229, 0.224, 0.225])

        pil_image = F.to_pil_image(image)

        open_cv_image = np.array(pil_image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        for x, y in keypoints:
                cv2.circle(open_cv_image, (x,y), radius=2, color=(0,0,255), thickness=-1)

        cv2.imshow("Augmentented", open_cv_image)
        cv2.waitKey(0)
        
        return None

    