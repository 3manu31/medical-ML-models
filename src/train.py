import os
import json
import argparse
import torch
import numpy as np

from data import get_data_loaders
from models import get_model
from losses import get_loss_criterion
from metrics import compute_metrics
from utils import set_seed, load_yaml_config, setup_logger, plot_learning_curves

def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    """
    Runs a single epoch of training or validation.
    """
    if is_train:
        model.train()
    else:
        model.eval()
        
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            # MedMNIST targets are sometimes shape [N, 1], flatten them to [N]
            targets = targets.squeeze().long()
            
            # Send data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, targets)
            
            # Backward pass & optimize if training
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Log metrics
            running_loss += loss.item() * images.size(0)
            
            # Probabilities (Softmax) & predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_probabilities.append(probs.detach().cpu().numpy())
            
    # Concatenate targets and predictions across batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_prob = np.concatenate(all_probabilities)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_metrics = compute_metrics(y_true, y_pred, y_prob)
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics

def main():
    parser = argparse.ArgumentParser(description="RetinaMNIST Clinical Decision Support Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save checkpoints and logs")
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml_config(args.config)
    model_name = config['model']['name']
    
    # Create specific output sub-directory for this model
    run_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Setup logger and reproducibility
    logger = setup_logger(run_output_dir)
    set_seed(config['training'].get('seed', 42))
    
    logger.info(f"Loaded config from: {args.config}")
    logger.info(f"Running experiments. Logs and output weights saved in: {run_output_dir}")
    
    # Device Configuration: Support Apple Silicon (mps), CUDA, or fallback CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon (MPS) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using NVIDIA CUDA acceleration.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")
        
    # Get loaders and class weights
    batch_size = config['training'].get('batch_size', 32)
    num_workers = config['training'].get('num_workers', 2)
    train_loader, val_loader, _, class_weights = get_data_loaders(batch_size=batch_size, num_workers=num_workers)
    
    logger.info(f"RetinaMNIST Dataloaders loaded. Dynamic class weights computed: {class_weights.tolist()}")
    
    # Model configuration
    dropout_rate = config['model'].get('dropout_rate', 0.4)
    model = get_model(model_name, dropout_rate=dropout_rate).to(device)
    logger.info(f"Model {model_name.upper()} initialized and moved to {device}.")
    
    # Criterion & Optimizer
    criterion = get_loss_criterion(class_weights=class_weights, device=device)
    
    lr = config['training'].get('learning_rate', 0.001)
    weight_decay = config['training'].get('weight_decay', 0.0001)
    
    if config['training'].get('optimizer', 'adam').lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
    # Learning Rate Scheduler
    # Reduces LR by 0.5 if Validation QWK does not improve for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )
        
    # Training Loop
    epochs = config['training'].get('epochs', 30)
    best_val_qwk = -1.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_qwk': [], 'val_qwk': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1': [], 'val_f1': []
    }
    
    for epoch in range(1, epochs + 1):
        # Get current learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        logger.info(f"--- Epoch {epoch}/{epochs} (Learning Rate: {curr_lr:.6f}) ---")
        
        # Train
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        # Validate
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)
        
        # Update Scheduler based on Validation QWK
        scheduler.step(val_metrics['qwk'])
        
        # Log to History
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_qwk'].append(train_metrics['qwk'])
        history['val_qwk'].append(val_metrics['qwk'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_macro'])
        history['val_f1'].append(val_metrics['f1_macro'])
        
        # Logging prints
        logger.info(
            f"Train -> Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | QWK: {train_metrics['qwk']:.4f}"
        )
        logger.info(
            f"Val   -> Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | QWK: {val_metrics['qwk']:.4f} | AUC-ROC: {val_metrics['auc_roc']:.4f}"
        )
        
        # Save checkpoints
        last_ckpt_path = os.path.join(run_output_dir, "last_model.pth")
        torch.save(model.state_dict(), last_ckpt_path)
        
        # Save best model based on validation QWK
        if val_metrics['qwk'] > best_val_qwk:
            best_val_qwk = val_metrics['qwk']
            best_ckpt_path = os.path.join(run_output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"New best validation QWK: {best_val_qwk:.4f}! Model checkpoint saved.")
            
    # Save training history JSON
    history_path = os.path.join(run_output_dir, "history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    logger.info("Training complete. Saving curves...")
    plot_learning_curves(history, run_output_dir)
    logger.info(f"Done! Results and checkpoints saved under: {run_output_dir}")

if __name__ == "__main__":
    main()
