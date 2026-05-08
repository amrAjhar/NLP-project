"""
Model evaluation module for AI-Generated Text Detection project.
Provides comprehensive evaluation metrics, confusion matrices, and ROC curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (14, 6)
        plt.rcParams['font.size'] = 10
    
    def compute_metrics(self, y_true, y_pred, y_scores=None, model_name="Model"):
        """
        Compute comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores for ROC (optional)
            model_name: Name of model for display
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
        
        if y_scores is not None:
            metrics['auc'] = roc_auc_score(y_true, y_scores)
        
        return metrics
    
    def create_comparison_table(self, metrics_list):
        """
        Create comparison table from multiple model metrics.
        
        Args:
            metrics_list: List of metric dictionaries
        
        Returns:
            DataFrame with comparison table
        """
        df = pd.DataFrame(metrics_list)
        return df
    
    def plot_confusion_matrices(self, model_results, y_true, save_path=None):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            model_results: List of tuples (model_name, y_pred)
            y_true: True labels
            save_path: Path to save figure (optional)
        """
        n_models = len(model_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, y_pred) in enumerate(model_results):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'],
                        ax=axes[idx], cbar=False)
            axes[idx].set_title(name, fontweight='bold', fontsize=12)
            axes[idx].set_ylabel('True')
            axes[idx].set_xlabel('Predicted')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, model_results, y_true, save_path=None):
        """
        Plot ROC curves for multiple models.
        
        Args:
            model_results: List of tuples (model_name, y_scores)
            y_true: True labels
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#1f77b4', '#d62728', '#9467bd', '#2ca02c', '#ff7f0e']
        
        for idx, (name, y_scores) in enumerate(model_results):
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})',
                    color=colors[idx % len(colors)])
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names, coefficients, top_n=20, save_path=None):
        """
        Plot feature importance from linear model.
        
        Args:
            feature_names: Array of feature names
            coefficients: Coefficient values
            top_n: Number of top features to show (default: 20)
            save_path: Path to save figure (optional)
        """
        top_indices = np.argsort(coefficients)[-top_n:][::-1]
        top_features = feature_names[top_indices]
        top_coeffs = coefficients[top_indices]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_coeffs, y=top_features, palette='Reds_r')
        plt.title('Top 20 Words Indicating AI-Generated Text', fontsize=14, fontweight='bold')
        plt.xlabel('SVM Coefficient Value', fontsize=12)
        plt.ylabel('Feature (Word/Bigram)', fontsize=12)
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        plt.show()
        
        print(f"\nTop {top_n} Words Indicating AI-Generated Text:")
        for i, (word, coeff) in enumerate(zip(top_features, top_coeffs), 1):
            print(f"  {i:2d}. {word:20s} ({coeff:.4f})")
    
    def plot_class_distribution(self, y, split_name="Dataset", save_path=None):
        """
        Plot class distribution.
        
        Args:
            y: Labels
            split_name: Name of data split (default: "Dataset")
            save_path: Path to save figure (optional)
        """
        unique, counts = np.unique(y, return_counts=True)
        labels = ['Human', 'AI']
        
        plt.figure(figsize=(8, 6))
        colors = ['#3498db', '#e74c3c']
        plt.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for i, (label, count) in enumerate(zip(labels, counts)):
            pct = count / len(y) * 100
            plt.text(i, count, f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.title(f'{split_name} Class Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Texts', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        
        plt.show()
    
    def print_metrics(self, metrics):
        """
        Pretty print metrics.
        
        Args:
            metrics: Dictionary of metrics or DataFrame
        """
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key != 'model':
                    print(f"  {key:>10}: {value:.4f}")
        elif isinstance(metrics, pd.DataFrame):
            print(metrics.to_string(index=False))


def create_summary_report(models_metrics, y_train, y_val, y_test):
    """
    Create a comprehensive summary report.
    
    Args:
        models_metrics: List of metric dictionaries
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
    """
    print(f"\n{'='*70}")
    print("FINAL COMPARISON: ALL MODELS")
    print(f"{'='*70}\n")
    
    comparison_df = pd.DataFrame(models_metrics)
    print(comparison_df.to_string(index=False))
    
    best_idx = comparison_df['f1'].idxmax()
    print(f"\n🏆 Best Overall Model (F1): {comparison_df.loc[best_idx, 'model']}")
    print(f"   F1-Score: {comparison_df.loc[best_idx, 'f1']:.4f}")
    
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}\n")
    
    print(f"Training set: {len(y_train):,} samples")
    train_dist = np.unique(y_train, return_counts=True)
    print(f"  Human: {train_dist[1][0]:,} ({train_dist[1][0]/len(y_train)*100:.1f}%)")
    print(f"  AI:    {train_dist[1][1]:,} ({train_dist[1][1]/len(y_train)*100:.1f}%)")
    
    print(f"\nValidation set: {len(y_val):,} samples")
    val_dist = np.unique(y_val, return_counts=True)
    print(f"  Human: {val_dist[1][0]:,} ({val_dist[1][0]/len(y_val)*100:.1f}%)")
    print(f"  AI:    {val_dist[1][1]:,} ({val_dist[1][1]/len(y_val)*100:.1f}%)")
    
    print(f"\nTest set: {len(y_test):,} samples")
    test_dist = np.unique(y_test, return_counts=True)
    print(f"  Human: {test_dist[1][0]:,} ({test_dist[1][0]/len(y_test)*100:.1f}%)")
    print(f"  AI:    {test_dist[1][1]:,} ({test_dist[1][1]/len(y_test)*100:.1f}%)")
    
    print(f"\n✅ Project complete! Ready for report writing.")
