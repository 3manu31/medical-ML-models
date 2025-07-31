import os
import random
import logging
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        # Metal Performance Shaders seed
        torch.mps.manual_seed(seed)

def load_yaml_config(config_path):
    """
    Load training configurations from a YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logger(output_dir):
    """
    Sets up file and console loggers.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('RetinaMNIST')

def plot_learning_curves(history, output_dir):
    """
    Generates and saves the loss and metrics learning curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss Curves (Epoch vs. Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # Plot QWK Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_qwk'], label='Train QWK', marker='s', color='green')
    plt.plot(epochs, history['val_qwk'], label='Val QWK', marker='s', color='orange')
    plt.title('Quadratic Weighted Kappa (QWK) Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Cohen\'s Kappa')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qwk_curve.png'))
    plt.close()
