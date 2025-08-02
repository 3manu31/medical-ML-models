import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from medmnist import RetinaMNIST
import torchvision.transforms as transforms

from models import get_model
from gradcam import GradCAM, generate_gradcam_overlay

def main():
    parser = argparse.ArgumentParser(description="Resolution Ablation Study: 28x28 vs 224x224 Grad-CAM")
    parser.add_argument("--cnn_checkpoint", type=str, default="outputs/cnn/best_model.pth", help="Path to 28x28 CNN checkpoint")
    parser.add_argument("--resnet_checkpoint", type=str, default="outputs/resnet18/best_model.pth", help="Path to 224x224 ResNet18 checkpoint")
    parser.add_argument("--output_path", type=str, default="assets/resolution_comparison.png", help="Path to save output comparison plot")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 1. Load Datasets
    # Transforms for 28x28
    transform_28 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Transforms for 224x224
    transform_224 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load same test split at both resolutions
    dataset_28 = RetinaMNIST(split="test", transform=transform_28, download=True, size=28)
    dataset_224 = RetinaMNIST(split="test", transform=transform_224, download=True, size=224)
    
    # Find the first 'Severe DR' (Class 3) sample index
    target_idx = None
    for idx in range(len(dataset_28)):
        label = dataset_28.labels[idx][0]
        if label == 3:  # Severe DR
            target_idx = idx
            break
            
    if target_idx is None:
        raise ValueError("Could not find a sample with Class 3 (Severe DR) in test set.")
        
    print(f"Found Severe DR (Class 3) sample at test index: {target_idx}")
    
    # Get images
    img_28, label_28 = dataset_28[target_idx]
    img_224, label_224 = dataset_224[target_idx]
    
    # Prepare batch dimensions [1, C, H, W]
    x_28 = img_28.unsqueeze(0)
    x_224 = img_224.unsqueeze(0)
    
    # 2. Initialize Models on CPU (safest for cross-resolution gradcam logic)
    # CNN Model (28x28)
    cnn_model = get_model("cnn", num_classes=5).cpu()
    cnn_model.load_state_dict(torch.load(args.cnn_checkpoint, map_location="cpu"))
    cnn_model.eval()
    
    # ResNet18 Model (224x224)
    resnet_model = get_model("resnet18", num_classes=5).cpu()
    resnet_model.load_state_dict(torch.load(args.resnet_checkpoint, map_location="cpu"))
    resnet_model.eval()
    
    # 3. Generate Grad-CAM for 28x28 CNN
    # Last conv layer for cnn: cnn_model.features[14]
    gradcam_28 = GradCAM(cnn_model, cnn_model.features[14])
    heatmap_28, _ = gradcam_28(x_28, class_idx=3)
    gradcam_28.remove_hooks()
    
    # Denormalize 28x28 image for visualization
    orig_28 = img_28.permute(1, 2, 0).numpy()
    orig_28 = (orig_28 * 0.5) + 0.5
    orig_28 = np.clip(orig_28, 0, 1)
    
    # Blend 28x28
    blended_28, _ = generate_gradcam_overlay(orig_28, heatmap_28, alpha=0.5)
    
    # 4. Generate Grad-CAM for 224x224 ResNet18
    # Last conv block layer for resnet18: resnet_model.resnet.layer4[-1]
    gradcam_224 = GradCAM(resnet_model, resnet_model.resnet.layer4[-1])
    heatmap_224, _ = gradcam_224(x_224, class_idx=3)
    gradcam_224.remove_hooks()
    
    # Denormalize 224x224 image for visualization
    orig_224 = img_224.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_224 = (orig_224 * std) + mean
    orig_224 = np.clip(orig_224, 0, 1)
    
    # Blend 224x224
    blended_224, _ = generate_gradcam_overlay(orig_224, heatmap_224, alpha=0.5)
    
    # 5. Plot and save side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left: 28x28 Baseline
    axes[0].imshow(blended_28)
    axes[0].set_title("28x28 Baseline (CNN)\nDiffuse, broad attention", fontsize=11)
    axes[0].axis("off")
    
    # Right: 224x224 ResNet18
    axes[1].imshow(blended_224)
    axes[1].set_title("224x224 Clinical Prototype (ResNet18)\nFocal, localized pathology attention", fontsize=11)
    axes[1].axis("off")
    
    plt.suptitle("Impact of Resolution on Pathological Localization\n(Patient with Severe Diabetic Retinopathy)", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200)
    plt.close()
    
    print(f"Comparison image successfully generated at {args.output_path}!")

if __name__ == "__main__":
    main()
