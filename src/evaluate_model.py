"""
Model evaluation module for AI-Generated Text Detection.
Includes metrics computation, visualization, and explainability (LIME, SHAP).
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, roc_auc_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
    
    def compute_metrics(self, y_true, y_pred, y_pred_proba=None, model_name='Model'):
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional, for AUC)
            model_name: Model name for storage
            
        Returns:
            Dict of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # AUC if probabilities provided
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        self.metrics[model_name] = metrics
        
        print(f"\n{'='*60}")
        print(f"METRICS - {model_name}")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred, target_names=['Human', 'AI'], digits=4))
        print(f"{'='*60}\n")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_dir='results/figures'):
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name
            save_dir: Directory to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Human', 'AI'],
                    yticklabels=['Human', 'AI'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, save_dir='results/figures', ax=None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Model name
            save_dir: Directory to save figure
            ax: Matplotlib axis (if None, creates new figure)
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        
        return fpr, tpr, roc_auc
    
    def plot_all_roc_curves(self, results_dict, save_dir='results/figures'):
        """
        Plot ROC curves for multiple models.
        
        Args:
            results_dict: Dict of {model_name: {'y_true': ..., 'y_pred_proba': ...}}
            save_dir: Directory to save figure
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, results in results_dict.items():
            self.plot_roc_curve(results['y_true'], results['y_pred_proba'], 
                               model_name, ax=plt.gca())
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'roc_curves_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
    
    def plot_metrics_comparison(self, save_dir='results/figures'):
        """
        Plot bar chart comparing metrics across models.
        
        Args:
            save_dir: Directory to save figure
        """
        models = list(self.metrics.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics_names):
            values = [self.metrics[model].get(metric, 0) for model in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
    
    def get_metrics_table(self):
        """Get metrics as a formatted table."""
        import pandas as pd
        
        df = pd.DataFrame(self.metrics).T
        df = df.round(4)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        print(df.to_string())
        print("="*80 + "\n")
        
        return df


class ExplainabilityAnalyzer:
    """Explainability analysis using LIME and SHAP."""
    
    @staticmethod
    def analyze_misclassifications(texts, y_true, y_pred, model_name='Model'):
        """
        Analyze and categorize misclassifications.
        
        Args:
            texts: List of texts
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name
            
        Returns:
            Dict with FP and FN examples
        """
        false_positives = []
        false_negatives = []
        
        for text, true, pred in zip(texts, y_true, y_pred):
            if true == 0 and pred == 1:  # Human misclassified as AI
                false_positives.append(text)
            elif true == 1 and pred == 0:  # AI misclassified as human
                false_negatives.append(text)
        
        print(f"\n{'='*60}")
        print(f"MISCLASSIFICATION ANALYSIS - {model_name}")
        print(f"{'='*60}")
        print(f"False Positives (Human → AI): {len(false_positives)}")
        print(f"False Negatives (AI → Human): {len(false_negatives)}")
        
        # Show examples
        print(f"\nTop 3 False Positives (misclassified as AI):")
        for i, text in enumerate(false_positives[:3], 1):
            print(f"\n{i}. {text[:150]}...")
        
        print(f"\n\nTop 3 False Negatives (misclassified as Human):")
        for i, text in enumerate(false_negatives[:3], 1):
            print(f"\n{i}. {text[:150]}...")
        
        print(f"\n{'='*60}\n")
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    @staticmethod
    def extract_lime_explanations(texts, model_predict_fn, num_samples=100, 
                                  num_words=10, num_examples=3):
        """
        Generate LIME explanations for misclassified samples.
        
        Args:
            texts: List of texts to explain
            model_predict_fn: Model prediction function
            num_samples: Number of perturbed samples for LIME
            num_words: Number of words to highlight
            num_examples: Number of examples to show
            
        Returns:
            List of explanations
        """
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            print("LIME not installed. Install with: pip install lime")
            return None
        
        print("\nGenerating LIME explanations...")
        explainer = LimeTextExplainer(class_names=['Human', 'AI'])
        
        explanations = []
        for i, text in enumerate(tqdm(texts[:num_examples])):
            exp = explainer.explain_instance(text, model_predict_fn, num_features=num_words)
            explanations.append({
                'text': text,
                'explanation': exp,
                'weights': dict(exp.as_list())
            })
        
        return explanations
    
    @staticmethod
    def extract_shap_feature_importance(texts, feature_extractor, model, 
                                       num_samples=100):
        """
        Extract SHAP feature importance.
        
        Args:
            texts: List of texts
            feature_extractor: Feature extraction function
            model: Trained model with predict method
            num_samples: Number of samples for SHAP computation
            
        Returns:
            SHAP explanations
        """
        try:
            import shap
        except ImportError:
            print("SHAP not installed. Install with: pip install shap")
            return None
        
        print("\nComputing SHAP values...")
        
        # Extract features
        X = feature_extractor(texts[:num_samples])
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(model.predict_proba, X[:50])
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X)
        
        return {
            'shap_values': shap_values,
            'X': X,
            'texts': texts[:num_samples]
        }
    
    @staticmethod
    def plot_error_analysis(false_positives, false_negatives, save_dir='results/figures'):
        """
        Plot error analysis statistics.
        
        Args:
            false_positives: List of FP texts
            false_negatives: List of FN texts
            save_dir: Directory to save figure
        """
        from nltk.tokenize import word_tokenize
        
        # Analyze text lengths
        fp_lengths = [len(word_tokenize(text)) for text in false_positives]
        fn_lengths = [len(word_tokenize(text)) for text in false_negatives]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Length distributions
        axes[0].hist(fp_lengths, bins=20, alpha=0.7, label='False Positives (Human→AI)', color='red')
        axes[0].hist(fn_lengths, bins=20, alpha=0.7, label='False Negatives (AI→Human)', color='blue')
        axes[0].set_xlabel('Text Length (tokens)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Text Length Distribution in Misclassifications', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Error counts
        error_types = ['False Positives\n(Human→AI)', 'False Negatives\n(AI→Human)']
        error_counts = [len(false_positives), len(false_negatives)]
        axes[1].bar(error_types, error_counts, color=['red', 'blue'])
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Misclassification Error Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(error_counts):
            axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'error_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
