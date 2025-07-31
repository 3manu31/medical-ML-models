import torch
import torch.nn as nn

def get_loss_criterion(class_weights=None, device="cpu"):
    """
    Returns the loss function.
    Uses standard Multi-class Cross-Entropy Loss with optional class weights to handle imbalance.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    return nn.CrossEntropyLoss(weight=class_weights)
