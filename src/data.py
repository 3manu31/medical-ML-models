import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from medmnist import RetinaMNIST

class InMemoryRetinaMNIST(Dataset):
    """
    Wraps the MedMNIST RetinaMNIST dataset by loading all images and labels into RAM
    at initialization time using 224x224 high-resolution images.
    """
    def __init__(self, split, transform=None):
        # Initialize original dataset once to download/extract at 224x224 size
        base_dataset = RetinaMNIST(split=split, download=True, size=224)
        
        # Load all data into memory
        self.imgs = base_dataset.imgs.copy()  # Shape: (N, 224, 224, 3)
        self.labels = base_dataset.labels.copy()  # Shape: (N, 1)
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        img = self.imgs[index]
        target = self.labels[index]
        
        # Apply transformation if specified
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target

def get_transforms():
    """
    Get training and validation/testing transforms with ImageNet normalizations
    and robust clinical augmentations.
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def get_class_weights(dataset):
    """
    Calculate class weights dynamically using the training dataset labels.
    """
    labels = dataset.labels.flatten()
    class_counts = np.bincount(labels)
    num_classes = len(class_counts)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.FloatTensor(class_weights)

def get_data_loaders(batch_size=32, num_workers=0):
    """
    Instantiate InMemoryRetinaMNIST datasets and return PyTorch DataLoaders.
    """
    train_transform, val_test_transform = get_transforms()
    
    # Load all splits completely into RAM once
    train_dataset = InMemoryRetinaMNIST(split="train", transform=train_transform)
    val_dataset = InMemoryRetinaMNIST(split="val", transform=val_test_transform)
    test_dataset = InMemoryRetinaMNIST(split="test", transform=val_test_transform)
    
    class_weights = get_class_weights(train_dataset)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights

if __name__ == "__main__":
    train_l, val_l, test_l, weights = get_data_loaders(batch_size=16)
    print(f"Train batches: {len(train_l)}")
    print(f"Val batches: {len(val_l)}")
    print(f"Test batches: {len(test_l)}")
    print(f"Computed Class Weights: {weights}")
    for images, labels in train_l:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break
