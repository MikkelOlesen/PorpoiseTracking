import os
import numpy as np
import torch
from PIL import Image

def load_keypoint_annotations(path, img_w, img_h):
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


class porpoise_keypoint_dataset(object):
    '''
        Creates a dataset with images and keypoints.
        
        Must have subfolders 'imgs', 'annotations' in the root dir. 
    '''

    def __init__(self, rootfolder, transforms):
        self.root = rootfolder
        self.transforms = transforms

        # load images and annotations
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "imgs"))))

    def __getitem__(self, idx):
        # load single image and and its annotations
        img_path = os.path.join(self.root, "imgs", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # get size of input image
        img_w, img_h = img.size

        #Load annotation file corresponding to image file
        annotation_name = os.path.splitext(self.imgs[idx])[0] + ".txt"
        annotations_path = os.path.join(self.root, "annotations", annotation_name)
        keypoints = load_keypoint_annotations(annotations_path, img_w, img_h)

        # convert keypoints into torch.Tensor
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)

        sample = {'image': img, 'keypoints': keypoints}

        #Apply image transforms
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    test = porpoise_keypoint_dataset("porpoise_keypoint_data", transforms=None)
    print(test.__getitem__(0))

