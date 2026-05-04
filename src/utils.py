"""
Utility functions for the AI-Generated Text Detection project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model (for title)
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        metrics_dict: Dict of {model_name: {metric: value}}
        save_path: Path to save figure (optional)
    """
    models = list(metrics_dict.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics_names):
        values = [metrics_dict[model].get(metric.lower().replace('-', '_'), 0) for model in models]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1])
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_metrics(y_true, y_pred, model_name):
    """
    Print classification metrics and confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"METRICS - {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, 
                                target_names=['Human', 'AI'],
                                digits=4))
    print(f"{'='*60}\n")


def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
