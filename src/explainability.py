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
    Native PyTorch implementation of Grad-CAM (Gradient-weighted Class Activation Mapping).
    Registers hooks on target convolutional layer to extract gradients and activations.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        # using register_full_backward_hook to avoid deprecation warnings
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
        # Shape of self.activations: [1, C, H, W]
        # Shape of self.gradients: [1, C, H, W]
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # GAP (Global Average Pooling) of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # ReLU of heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        denom = np.max(heatmap) - np.min(heatmap)
        if denom > 0:
            heatmap = (heatmap - np.min(heatmap)) / denom
            
        return heatmap, class_idx

def generate_and_save_gradcam(image_tensor, original_image, heatmap, class_idx, class_name, output_path):
    """
    Overlays Grad-CAM heatmap onto the original image and saves the plot.
    """
    # Resize heatmap to match image size (28x28)
    heatmap_resized = cv2.resize(heatmap, (28, 28))
    
    # Convert heatmap to jet colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Normalize original image to [0, 1] range for visualization
    orig_img_normalized = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-8)
    
    # Blend image and heatmap
    alpha = 0.5
    blended = (orig_img_normalized * (1.0 - alpha)) + ((heatmap_colored / 255.0) * alpha)
    
    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    # Original Image
    axes[0].imshow(orig_img_normalized)
    axes[0].set_title(f"Original (Class {class_idx})")
    axes[0].axis('off')
    
    # Blended Heatmap
    axes[1].imshow(blended)
    axes[1].set_title(f"Grad-CAM ({class_name})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Explainable Grad-CAM Heatmaps")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model weight checkpoint")
    parser.add_argument("--output_dir", type=str, default="./assets", help="Output directory to save assets")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_yaml_config(args.config)
    model_name = config['model']['name']
    if model_name.lower() != 'cnn':
        raise ValueError("Grad-CAM is only applicable to the Custom CNN model.")
        
    # Get loader
    _, _, test_loader, _ = get_data_loaders(batch_size=1)
    
    # Initialize and load model
    model = get_model(model_name).cpu()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    
    # Get target convolution layer (last conv layer in self.features is block[14])
    target_layer = model.features[14]
    grad_cam = GradCAM(model, target_layer)
    
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]
    found_classes = {}
    
    # Find one representative image from test split for each class
    for images, targets in test_loader:
        target_idx = targets.squeeze().item()
        
        # If we haven't processed this class yet
        if target_idx not in found_classes:
            # Denormalize image for plotting (from [-1, 1] back to original range)
            image_np = images[0].permute(1, 2, 0).numpy()
            image_np = (image_np * 0.5) + 0.5 # Undo normalization
            image_np = np.clip(image_np, 0, 1)
            
            # Save target image tensor & original image representation
            found_classes[target_idx] = {
                'tensor': images,
                'orig': image_np
            }
            
        if len(found_classes) == 5:
            break
            
    print("Generating Grad-CAM visualizations for each retinopathy severity level...")
    
    # Process each class
    for class_idx in sorted(found_classes.keys()):
        class_data = found_classes[class_idx]
        image_tensor = class_data['tensor']
        orig_image = class_data['orig']
        
        # Generate Grad-CAM
        heatmap, predicted_idx = grad_cam(image_tensor, class_idx=class_idx)
        
        # Save output image
        out_filename = f"gradcam_class_{class_idx}.png"
        out_path = os.path.join(args.output_dir, out_filename)
        
        generate_and_save_gradcam(
            image_tensor, 
            orig_image, 
            heatmap, 
            class_idx, 
            class_names[class_idx], 
            out_path
        )
        print(f"Saved Grad-CAM for {class_names[class_idx]} to {out_path}")
        
    # Remove hooks
    grad_cam.remove_hooks()
    print("Grad-CAM generation complete.")

if __name__ == "__main__":
    main()
