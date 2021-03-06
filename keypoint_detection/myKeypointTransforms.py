import random
import numpy as np
from torchvision.transforms import functional as F
import cv2
import torch
import math
from PIL import Image
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
            noise = torch.randn_like(image) * self.factor 
            image = image + noise
        return {'image': image, 'keypoints': keypoints}


class RandomFlip(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        value = random.random()

        if value <= 0.25:  #Horiensontal flip
            height, width = image.shape[-2:]
            image = F.hflip(image)

            for i, (x, y) in enumerate(keypoints):
                keypoints[i][0] = width - x

            temp_keypoints = keypoints.clone()
            keypoints[3] = temp_keypoints[2]
            keypoints[2] = temp_keypoints[3]
        elif(value <= 0.5): #Vertical flip
            height, width = image.shape[-2:]
            image = F.vflip(image)

            for i, (x, y) in enumerate(keypoints):
                keypoints[i][1] = height - y

            temp_keypoints = keypoints.clone()
            keypoints[3] = temp_keypoints[2]
            keypoints[2] = temp_keypoints[3]
        elif(value <= 0.75): #Double flip (rotate 180)
            height, width = image.shape[-2:]
            image = F.vflip(image)
            image = F.hflip(image)

            for i, (x, y) in enumerate(keypoints):
                keypoints[i][1] = height - y
                keypoints[i][0] = width - x
        else:
            pass
        return {'image': image, 'keypoints': keypoints}


class ShowImg(object):
    '''
    Shows image in opencv window. 
    Used to test transforms
    '''
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']


        pil_image = F.to_pil_image(image)

        open_cv_image = np.array(pil_image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        for i, (x, y) in enumerate(keypoints):
                t = i + 1
                cv2.circle(open_cv_image, (x,y), radius=2, color=(int(255/t),0,int((255/4) * t)), thickness=-1)

        cv2.imshow("Augmentented", open_cv_image)
        cv2.waitKey(0)
        
        return None

    