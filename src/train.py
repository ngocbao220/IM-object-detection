import torch
import os
from dataset import VOCDetectionDataset, collate_fn
from transforms import get_transforms
from model import get_model
from constants import NUM_CLASSES

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(f"Epoch: [{epoch}] Batch: [{i}/{len(data_loader)}] Loss: {losses.item():.4f}")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Dataset paths
    root = "." # Points to where VOCdevkit is
    
    dataset = VOCDetectionDataset(root, year='2012', image_set='train', transforms=get_transforms(train=True))
    dataset_test = VOCDetectionDataset(root, year='2012', image_set='val', transforms=get_transforms(train=False))

    # Subset for trial
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:20])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn)

    model = get_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 1 # Keep it 1 for demonstration/initial test
    
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        
        # Save model
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    print("Training complete.")

if __name__ == "__main__":
    main()
