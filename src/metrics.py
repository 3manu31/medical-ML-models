import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score

def compute_metrics(y_true, y_pred, y_prob):
    """
    Computes diagnostic metrics for evaluation.
    
    Args:
        y_true (np.ndarray): Ground truth labels (shape: [num_samples])
        y_pred (np.ndarray): Predicted classes (shape: [num_samples])
        y_prob (np.ndarray): Softmax probability scores (shape: [num_samples, num_classes])
        
    Returns:
        dict: Collection of accuracy, macro F1, ROC-AUC, and Quadratic Weighted Kappa.
    """
    metrics = {}
    
    # 1. Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 2. Macro F1-score
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 3. Quadratic Weighted Cohen's Kappa (QWK)
    # Essential for Retinal severity grading
    metrics['qwk'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # 4. AUC-ROC (One-vs-Rest)
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception as e:
        # Fallback if some classes have 0 samples in evaluation set
        metrics['auc_roc'] = 0.0
        
    return metrics
