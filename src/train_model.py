"""
Model training module for AI-Generated Text Detection.
Implements training loops for Logistic Regression, SVM, BERT, and Ensemble models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import joblib
from datetime import datetime


class LogisticRegressionTrainer:
    """Logistic Regression model trainer."""
    
    def __init__(self, max_iter=1000, random_state=42):
        """
        Initialize Logistic Regression.
        
        Args:
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state, n_jobs=-1)
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Logistic Regression.
        
        Args:
            X_train: Training features (dense array or sparse matrix)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dict with training info
        """
        print("Training Logistic Regression...")
        
        # Convert sparse to dense if needed
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set if provided
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = None
        
        if X_val is not None and y_val is not None:
            if hasattr(X_val, 'toarray'):
                X_val = X_val.toarray()
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
        
        print(f"Training accuracy: {train_score:.4f}")
        if val_score:
            print(f"Validation accuracy: {val_score:.4f}")
        
        return {'train_score': train_score, 'val_score': val_score}
    
    def predict(self, X):
        """Make predictions."""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, path):
        """Save model."""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load(self, path):
        """Load model."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']


class SVMTrainer:
    """SVM model trainer."""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        """
        Initialize SVM.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
            random_state: Random seed
        """
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, 
                         probability=True, random_state=random_state)
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train SVM.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dict with training info
        """
        print("Training SVM...")
        
        # Convert sparse to dense
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = None
        
        if X_val is not None and y_val is not None:
            if hasattr(X_val, 'toarray'):
                X_val = X_val.toarray()
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
        
        print(f"Training accuracy: {train_score:.4f}")
        if val_score:
            print(f"Validation accuracy: {val_score:.4f}")
        
        return {'train_score': train_score, 'val_score': val_score}
    
    def predict(self, X):
        """Make predictions."""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, path):
        """Save model."""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load(self, path):
        """Load model."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']


class BERTClassifier(nn.Module):
    """BERT-based text classification model."""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.1):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class BERTTrainer:
    """BERT model trainer."""
    
    def __init__(self, model_name='bert-base-uncased', device=None, learning_rate=2e-5):
        """
        Initialize BERT trainer.
        
        Args:
            model_name: HuggingFace model name
            device: Torch device
            learning_rate: Learning rate for fine-tuning
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BERTClassifier(model_name=model_name).to(self.device)
    
    def tokenize_texts(self, texts, max_length=512):
        """
        Tokenize texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Tokenized inputs
        """
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        return encodings
    
    def train(self, texts_train, labels_train, texts_val=None, labels_val=None,
              epochs=3, batch_size=16, warmup_steps=0):
        """
        Fine-tune BERT.
        
        Args:
            texts_train: Training texts
            labels_train: Training labels
            texts_val: Validation texts (optional)
            labels_val: Validation labels (optional)
            epochs: Number of epochs
            batch_size: Batch size
            warmup_steps: Number of warmup steps
            
        Returns:
            Dict with training history
        """
        print(f"Fine-tuning BERT for {epochs} epochs on {self.device}...")
        
        # Tokenize training data
        train_encodings = self.tokenize_texts(texts_train)
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(labels_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Tokenize validation data if provided
        val_loader = None
        if texts_val is not None and labels_val is not None:
            val_encodings = self.tokenize_texts(texts_val)
            val_dataset = TensorDataset(
                val_encodings['input_ids'],
                val_encodings['attention_mask'],
                torch.tensor(labels_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss = self._evaluate(val_loader, loss_fn)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
        
        return history
    
    def _evaluate(self, data_loader, loss_fn):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def predict(self, texts, batch_size=32):
        """Make predictions."""
        self.model.eval()
        encodings = self.tokenize_texts(texts)
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        loader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        
        return np.array(predictions)
    
    def predict_proba(self, texts, batch_size=32):
        """Get prediction probabilities."""
        self.model.eval()
        encodings = self.tokenize_texts(texts)
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
        loader = DataLoader(dataset, batch_size=batch_size)
        
        probabilities = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.extend(probs)
        
        return np.array(probabilities)
    
    def save(self, path):
        """Save model."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Load model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
