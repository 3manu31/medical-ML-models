import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from data import get_data_loaders
from models import get_model
from metrics import compute_metrics
from utils import load_yaml_config

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """
    Plots and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Row normalization to show sensitivity per class
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_counts.png'))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix (Recall / Sensitivity)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_normalized.png'))
    plt.close()

def plot_roc_curves(y_true, y_prob, num_classes, class_names, output_path):
    """
    Plots and saves multi-class One-vs-Rest ROC curves.
    """
    plt.figure(figsize=(8, 6))
    
    # Compute ROC curve and ROC area for each class
    for i in range(num_classes):
        # Create binary labels for the current class
        y_true_binary = (y_true == i).astype(int)
        
        # Check if class exists in targets to prevent errors
        if np.sum(y_true_binary) > 0:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="RetinaMNIST Clinical Decision Support Evaluation Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save evaluation artifacts")
    args = parser.parse_args()
    
    # Load configurations
    config = load_yaml_config(args.config)
    model_name = config['model']['name']
    
    # Create specific output directories
    eval_output_dir = os.path.join(args.output_dir, model_name, "evaluation")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device {device} for evaluation.")
    
    # Get loader
    _, _, test_loader, _ = get_data_loaders(batch_size=config['training'].get('batch_size', 32))
    
    # Initialize and load model weights
    dropout_rate = config['model'].get('dropout_rate', 0.4)
    model = get_model(model_name, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            targets = targets.squeeze().long()
            images = images.to(device)
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_probabilities.append(probs.cpu().numpy())
            
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_prob = np.concatenate(all_probabilities)
    
    # Calculate general metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    print("\n" + "="*40)
    print(f"Evaluation Metrics on Test Split ({model_name.upper()}):")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {metrics['qwk']:.4f}")
    print(f"Macro AUC-ROC: {metrics['auc_roc']:.4f}")
    print("="*40 + "\n")
    
    # Save metrics JSON
    with open(os.path.join(eval_output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Generate markdown table for metrics
    md_summary = f"""# Test Set Performance Summary - {model_name.upper()}

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | {metrics['accuracy']:.4f} | Overall ratio of correctly predicted samples. |
| **Macro F1** | {metrics['f1_macro']:.4f} | Class-balanced harmonic mean of precision and recall. |
| **QWK** | {metrics['qwk']:.4f} | Quadratic Weighted Kappa (Clinical diagnostic agreement score). |
| **AUC-ROC** | {metrics['auc_roc']:.4f} | ROC Area Under Curve (Ability to distinguish between severity levels). |
"""
    with open(os.path.join(eval_output_dir, "test_metrics_summary.md"), "w") as f:
        f.write(md_summary)
        
    # Generate plots
    # RetinaMNIST classes
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Proliferative"]
    
    print("Generating plots...")
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(eval_output_dir, "confusion_matrix.png"))
    plot_roc_curves(y_true, y_prob, len(class_names), class_names, os.path.join(eval_output_dir, "roc_curve.png"))
    
    print(f"Evaluation complete. Artifacts saved in: {eval_output_dir}")

if __name__ == "__main__":
    main()
