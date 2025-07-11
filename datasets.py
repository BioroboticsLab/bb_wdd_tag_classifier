from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

from hyperparameters import batch_size

train_transform = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ]
)

train_dataset = datasets.ImageFolder(
    Path.cwd() / "data" / "cropped" / "50x50" / "train",
    train_transform,
)
validation_dataset = datasets.ImageFolder(
    Path.cwd() / "data" / "cropped" / "50x50" / "validation",
    test_transform,
)
test_dataset = datasets.ImageFolder(
    Path.cwd() / "data" / "cropped" / "50x50" / "test",
    test_transform,
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
)
test_dataloader = DataLoader(test_dataset, batch_size=120)

if __name__ == "__main__":
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
