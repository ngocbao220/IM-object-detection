import torch
from torchvision.datasets import VOCDetection
from constants import CLASS_TO_IDX

class VOCDetectionDataset(VOCDetection):
    def __init__(self, root, year, image_set, transforms=None):
        super().__init__(root, year=year, image_set=image_set, download=False)
        self._transforms = transforms

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        
        # Target from VOCDetection is a dictionary reflecting XML structure.
        # We need to extract boxes and labels for Faster R-CNN.
        boxes = []
        labels = []
        
        objs = target['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
            
        for obj in objs:
            name = obj['name']
            labels.append(CLASS_TO_IDX[name])
            
            bndbox = obj['bndbox']
            # Coordinates are 1-based in VOC, convert to 0-based
            xmin = float(bndbox['xmin']) - 1
            ymin = float(bndbox['ymin']) - 1
            xmax = float(bndbox['xmax']) - 1
            ymax = float(bndbox['ymax']) - 1
            boxes.append([xmin, ymin, xmax, ymax])
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowded = torch.zeros((len(objs),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowded"] = iscrowded
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))
