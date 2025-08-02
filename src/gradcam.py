import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from data import get_data_loaders
from models import get_model
from utils import load_yaml_config

class GradCAM:
    """
    Grad-CAM class designed to register forward/backward hooks on target conv layers,
    allowing diagnostic explainability by tracking gradient flow and activation heatmaps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0][class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Extract features and gradients
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # GAP of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # ReLU & Normalization
        heatmap = np.maximum(heatmap, 0)
        denom = np.max(heatmap) - np.min(heatmap)
        if denom > 0:
            heatmap = (heatmap - np.min(heatmap)) / denom
            
        return heatmap, class_idx

def generate_gradcam_overlay(original_image, heatmap, alpha=0.5):
    """
    Resizes the heatmap to match the original image dimensions, applies a colormap,
    and blends it with the original image.
    """
    # Resize heatmap to match original image (224x224)
    height, width, _ = original_image.shape
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Convert heatmap to color
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Normalize original image to [0, 1] range for visual blend
    orig_img_norm = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-8)
    
    # Blend
    blended = (orig_img_norm * (1.0 - alpha)) + ((heatmap_colored / 255.0) * alpha)
    blended = np.clip(blended, 0, 1)
    
    return blended, orig_img_norm

def save_gradcam_plot(orig_img, blended_img, class_idx, class_name, output_path):
    """
    Plots the original image side-by-side with the Grad-CAM blended visual overlay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Original Image
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Original Fundus (Class {class_idx})")
    axes[0].axis('off')
    
    # Blended Heatmap
    axes[1].imshow(blended_img)
    axes[1].set_title(f"Grad-CAM (Focus Area: {class_name})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Clinical Model Explainability Generator")
    parser.add_argument("--config", type=str, default="configs/hero_resnet.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ResNet18 model weight checkpoint")
    parser.add_argument("--output_dir", type=str, default="./assets", help="Directory to save visual output")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_yaml_config(args.config)
    model_name = config['model']['name']
    
    # Get loader
    _, _, test_loader, _ = get_data_loaders(batch_size=1)
    
    # Load Model
    model = get_model(model_name).cpu()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    
    # Target last conv layer block in ResNet18
    target_layer = model.resnet.layer4[-1]
    
    grad_cam = GradCAM(model, target_layer)
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]
    found_classes = {}
    
    # Find one representative image from test split for each class
    for images, targets in test_loader:
        target_idx = targets.squeeze().item()
        
        if target_idx not in found_classes:
            # Denormalize image for plotting (from ImageNet normalization back to original range)
            image_np = images[0].permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = (image_np * std) + mean
            image_np = np.clip(image_np, 0, 1)
            
            found_classes[target_idx] = {
                'tensor': images,
                'orig': image_np
            }
            
        if len(found_classes) == 5:
            break
            
    print("Generating Grad-CAM overlays for ResNet18...")
    
    for class_idx in sorted(found_classes.keys()):
        class_data = found_classes[class_idx]
        image_tensor = class_data['tensor']
        orig_image = class_data['orig']
        
        heatmap, predicted_idx = grad_cam(image_tensor, class_idx=class_idx)
        blended, orig_norm = generate_gradcam_overlay(orig_image, heatmap, alpha=0.45)
        
        out_filename = f"gradcam_class_{class_idx}.png"
        out_path = os.path.join(args.output_dir, out_filename)
        
        save_gradcam_plot(orig_norm, blended, class_idx, class_names[class_idx], out_path)
        print(f"Saved Grad-CAM overlay for {class_names[class_idx]} to {out_path}")
        
    grad_cam.remove_hooks()
    print("Done generating explainability maps.")

if __name__ == "__main__":
    main()
