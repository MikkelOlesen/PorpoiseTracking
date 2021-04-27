import os
import numpy as np
import torch
from PIL import Image

def load_annotations(path, img_w, img_h):
    """
        Loads bboxes in annotation file at 'path' and rescales from 0-1 to 0 to w,h
        Also changes from xywh to x1y1x2y2
        Returns bboxes
    """
    bboxes = []
    with open(path, 'r') as file:
        for row in file:
            _, xc , yc, w, h = row.split()
            xc = float(xc)*img_w
            yc = float(yc)*img_h
            w = float(w)*img_w
            h = float(h)*img_h
            bboxes.append([xc - w/2 , yc - h/2, xc + w/2 , yc + h/2])

    return bboxes


class porpoise_dataset(object):
    '''
        Creates a dataset with images and bboxes.
        
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
        bboxes = load_annotations(annotations_path, img_w, img_h)
        num_bboxes = len(bboxes)

        # convert bboxes into a torch.Tensor
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)

        # convert labels into tensor, since there is only one class the tensor is just filled with ones
        labels = torch.ones((num_bboxes,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((num_bboxes,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    test = porpoise_dataset("porpoise_detection_data", transforms=None)
    print(test.__getitem__(725))

