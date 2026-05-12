import torch
from torchvision import transforms as T

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(torch.nn.Module):
    def forward(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target

def get_transforms(train=False):
    transforms = []
    transforms.append(ToTensor())
    # You can add more augmentations like RandomHorizontalFlip here if train=True
    return Compose(transforms)
