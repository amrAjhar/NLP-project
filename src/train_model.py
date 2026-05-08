"""
Model training module for AI-Generated Text Detection project.
Implements three classification models: Linear SVM, XGBoost, and Neural Network.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import xgboost as xgb


class LinearSVMTrainer:
    """
    Train a linear SVM using SGDClassifier (more efficient than RBF kernel).
    Loss='hinge' makes SGDClassifier mathematically equivalent to Linear SVM.
    """
    
    def __init__(self, alpha=1e-4, random_state=42, max_iter=1000, n_jobs=-1):
        """
        Initialize SVM trainer.
        
        Args:
            alpha: Regularization parameter (default: 1e-4)
            random_state: Random seed (default: 42)
            max_iter: Maximum iterations (default: 1000)
            n_jobs: Number of jobs for parallelization (default: -1, all cores)
        """
        self.model = SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=alpha,
            random_state=random_state,
            max_iter=max_iter,
            tol=1e-3,
            n_jobs=n_jobs
        )
    
    def train(self, X_train, y_train):
        """
        Train the SVM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n{'='*60}")
        print("MODEL 1: LINEAR SVM (via SGD - Zero Memory Overhead)")
        print(f"{'='*60}\n")
        
        print("Training SGD SVM (Streaming rows to save RAM)...")
        self.model.fit(X_train, y_train)
        print("✅ Training complete!")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
        
        Returns:
            Decision function scores (use for ROC curves)
        """
        return self.model.decision_function(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        y_scores = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_scores)
        }
        
        print(f"\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        return metrics


class XGBoostTrainer:
    """
    Train an XGBoost classifier.
    """
    
    def __init__(self, n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42):
        """
        Initialize XGBoost trainer.
        
        Args:
            n_estimators: Number of boosting rounds (default: 100)
            max_depth: Maximum tree depth (default: 7)
            learning_rate: Learning rate (default: 0.1)
            random_state: Random seed (default: 42)
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
    
    def train(self, X_train, y_train):
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n{'='*60}")
        print("MODEL 2: XGBOOST (GLOVE EMBEDDINGS)")
        print(f"{'='*60}\n")
        
        print("Training...")
        self.model.fit(X_train, y_train, verbose=False)
        print("✅ Training complete!")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
        
        Returns:
            Probability for positive class
        """
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        return metrics


class SimpleNN(nn.Module):
    """
    Simple feedforward neural network for DistilBERT embeddings.
    """
    
    def __init__(self, input_dim=768):
        """
        Initialize neural network.
        
        Args:
            input_dim: Input dimension (default: 768 for DistilBERT)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)
    
    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class NeuralNetworkTrainer:
    """
    Train a neural network on DistilBERT embeddings.
    """
    
    def __init__(self, input_dim=768, batch_size=256, learning_rate=0.001, num_epochs=5):
        """
        Initialize neural network trainer.
        
        Args:
            input_dim: Input dimension (default: 768)
            batch_size: Batch size (default: 256)
            learning_rate: Learning rate (default: 0.001)
            num_epochs: Number of training epochs (default: 5)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleNN(input_dim).to(self.device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, X_train, y_train):
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n{'='*60}")
        print("MODEL 3: NEURAL NETWORK (DISTILBERT)")
        print(f"{'='*60}\n")
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        print("Training neural network...")
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        print("✅ Training complete!")
    
    def predict(self, X_test):
        """
        Make predictions on new data.
        
        Args:
            X_test: Test features
        
        Returns:
            Predicted labels and probabilities
        """
        test_dataset = TensorDataset(torch.FloatTensor(X_test))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        predictions = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of metrics
        """
        y_pred, y_proba = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        return metrics
