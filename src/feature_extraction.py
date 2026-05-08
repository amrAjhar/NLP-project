"""
Feature extraction module for AI-Generated Text Detection project.
Implements three feature extraction methods: TF-IDF, GloVe, and DistilBERT.
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gensim.downloader as api
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import gc


class TFIDFFeatureExtractor:
    """
    Extract TF-IDF features from text documents.
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.90):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features (default: 5000)
            ngram_range: N-gram range (default: (1, 2))
            min_df: Minimum document frequency (default: 5)
            max_df: Maximum document frequency (default: 0.90)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            dtype=np.float32
        )
        self.scaler = StandardScaler(with_mean=False)
        self.fitted = False
    
    def fit_transform(self, X_train):
        """
        Fit vectorizer on training data and transform.
        
        Args:
            X_train: Training texts
        
        Returns:
            Scaled sparse matrix
        """
        print(f"\n{'='*60}")
        print("FEATURE METHOD 1: TF-IDF (Full Dataset)")
        print(f"{'='*60}\n")
        
        print(f"Fitting TF-IDF on {len(X_train):,} training texts...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train.astype(str))
        
        print("Scaling TF-IDF matrix...")
        X_train_tfidf_scaled = self.scaler.fit_transform(X_train_tfidf)
        
        self.fitted = True
        
        print(f"\nTF-IDF Results:")
        print(f"  Train: {X_train_tfidf_scaled.shape}")
        sparsity = 1 - X_train_tfidf_scaled.nnz / (X_train_tfidf_scaled.shape[0] * X_train_tfidf_scaled.shape[1])
        print(f"  Sparsity: {sparsity:.1%}")
        print(f"  Memory: {X_train_tfidf_scaled.data.nbytes / 1e6:.1f} MB")
        print(f"\n✅ TF-IDF extraction complete!")
        
        return X_train_tfidf_scaled
    
    def transform(self, X):
        """
        Transform new data using fitted vectorizer.
        
        Args:
            X: Text data to transform
        
        Returns:
            Scaled sparse matrix
        """
        if not self.fitted:
            raise ValueError("Vectorizer not fitted yet")
        X_tfidf = self.vectorizer.transform(X.astype(str))
        return self.scaler.transform(X_tfidf)


class GloVeFeatureExtractor:
    """
    Extract GloVe embeddings from text documents.
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize GloVe embeddings.
        
        Args:
            embedding_model: Pre-loaded GloVe model (loads if None)
        """
        print(f"\n{'='*60}")
        print("FEATURE METHOD 2: GLOVE EMBEDDINGS (Full Dataset)")
        print(f"{'='*60}\n")
        
        if embedding_model is None:
            print("Loading GloVe embeddings (300-dim)...")
            self.embedding_model = api.load('glove-wiki-gigaword-300')
            print(f"✅ Loaded (vector size: {self.embedding_model.vector_size})")
        else:
            self.embedding_model = embedding_model
        
        self.scaler = StandardScaler()
        self.fitted = False
    
    def _get_embedding_vector(self, text, aggregation='mean'):
        """
        Get embedding vector for a single text.
        
        Args:
            text: Input text
            aggregation: Aggregation method (default: 'mean')
        
        Returns:
            Embedding vector
        """
        words = text.split()
        embeddings = []
        for word in words:
            if word in self.embedding_model:
                embeddings.append(self.embedding_model[word])
        
        if embeddings:
            if aggregation == 'mean':
                return np.mean(embeddings, axis=0)
        return np.zeros(self.embedding_model.vector_size)
    
    def fit_transform(self, X_train):
        """
        Extract GloVe embeddings for training data.
        
        Args:
            X_train: Training texts
        
        Returns:
            Scaled embeddings
        """
        print(f"Extracting embeddings for train set ({len(X_train):,} texts)...")
        X_train_glove = np.array([
            self._get_embedding_vector(text)
            for text in tqdm(X_train, desc="Train")
        ])
        
        print("Scaling GloVe embeddings...")
        X_train_glove_scaled = self.scaler.fit_transform(X_train_glove)
        
        self.fitted = True
        
        print(f"\nGloVe Results:")
        print(f"  Train: {X_train_glove_scaled.shape}")
        print(f"  Memory: {X_train_glove_scaled.nbytes / 1e6:.1f} MB")
        print(f"\n✅ GloVe extraction complete!")
        gc.collect()
        
        return X_train_glove_scaled
    
    def transform(self, X):
        """
        Extract GloVe embeddings for new data.
        
        Args:
            X: Text data to transform
        
        Returns:
            Scaled embeddings
        """
        if not self.fitted:
            raise ValueError("Extractor not fitted yet")
        
        X_glove = np.array([
            self._get_embedding_vector(text)
            for text in X
        ])
        return self.scaler.transform(X_glove)


class DistilBERTFeatureExtractor:
    """
    Extract DistilBERT embeddings from text documents.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', batch_size=256, max_length=128):
        """
        Initialize DistilBERT model.
        
        Args:
            model_name: HuggingFace model name (default: 'distilbert-base-uncased')
            batch_size: Batch size for processing (default: 256)
            max_length: Maximum sequence length (default: 128)
        """
        print(f"\n{'='*60}")
        print("FEATURE METHOD 3: DISTILBERT (Batch Extraction)")
        print(f"{'='*60}\n")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        print(f"Loading {model_name} tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("✅ Model loaded")
        
        self.batch_size = batch_size
        self.max_length = max_length
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit_transform(self, X_train):
        """
        Extract DistilBERT embeddings for training data.
        
        Args:
            X_train: Training texts
        
        Returns:
            Scaled embeddings
        """
        print(f"\n1. Train set ({len(X_train):,} texts)...")
        X_train_bert = self._extract_batch(X_train)
        
        print("Scaling DistilBERT embeddings...")
        X_train_bert_scaled = self.scaler.fit_transform(X_train_bert)
        
        self.fitted = True
        
        print(f"\nDistilBERT Results:")
        print(f"  Train: {X_train_bert_scaled.shape}")
        print(f"  Memory: {X_train_bert_scaled.nbytes / 1e9:.2f} GB")
        print(f"\n✅ DistilBERT extraction complete!")
        
        del self.model
        gc.collect()
        
        return X_train_bert_scaled
    
    def _extract_batch(self, texts):
        """
        Extract embeddings in batches.
        
        Args:
            texts: List of texts
        
        Returns:
            Embeddings array
        """
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Batches", total=total_batches):
                batch = list(texts[i:min(i+self.batch_size, len(texts))])
                
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**encoded)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embedding)
        
        return np.array(embeddings)
    
    def transform(self, X):
        """
        Extract DistilBERT embeddings for new data.
        
        Args:
            X: Text data to transform
        
        Returns:
            Scaled embeddings
        """
        if not self.fitted:
            raise ValueError("Extractor not fitted yet")
        
        X_bert = self._extract_batch(X)
        return self.scaler.transform(X_bert)
