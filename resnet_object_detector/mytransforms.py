from torchvision.transforms import functional as F

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
        w, h = image.size
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