import os
import numpy as np
import torch
from PIL import Image

def load_keypoint_annotations(path, img_w, img_h):
    """
        Loads keypoints in annotation file at 'path'
        Returns bboxes
    """
    keypoints = []
    with open(path, 'r') as file:
        for row in file:
            x, y , v = row.split()
            keypoints.append([float(x), float(y), float(v)])

    return keypoints


class porpoise_keypoint_dataset(object):
    '''
        Creates a dataset with images and keypoints.
        
        Must have subfolders imgs, annotations in the root dir. 
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

        annotation_name = os.path.splitext(self.imgs[idx])[0] + ".txt"
        annotations_path = os.path.join(self.root, "annotations", annotation_name)
        keypoints = load_keypoint_annotations(annotations_path, img_w, img_h)

        # convert keypoints into torch.Tensor
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)

        # createing bbox that filles entire image. As imgs are only close-up imgs of porpoises
        bbox = [[0, 0, img_w, img_h]]

        # convert bboxes into a torch.Tensor
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        num_bboxes = len(bbox)

        # convert labels into tensor, since there is only one class the tensor is just filled with ones
        labels = torch.ones((num_bboxes,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_bboxes,), dtype=torch.int64)
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        

        target = {}
        target["boxes"] = bbox
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["keypoints"] = keypoints
        

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    test = porpoise_keypoint_dataset("porpoise_keypoint_data", transforms=None)
    print(test.__getitem__(0))

