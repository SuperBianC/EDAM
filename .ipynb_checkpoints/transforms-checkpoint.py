import random
import torch

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
