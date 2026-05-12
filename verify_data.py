from src.dataset import VOCDetectionDataset
from src.transforms import get_transforms
import torch

def verify_data():
    try:
        dataset = VOCDetectionDataset(root=".", year='2012', image_set='train', transforms=get_transforms(train=False))
        print(f"Dataset size: {len(dataset)}")
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Target keys: {target.keys()}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Labels: {target['labels']}")
        print("Data verification successful!")
    except Exception as e:
        print(f"Data verification failed: {e}")

if __name__ == "__main__":
    verify_data()
