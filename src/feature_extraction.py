"""
Feature extraction module for converting text to numerical representations.
Implements TF-IDF, word embeddings (Word2Vec/GloVe), and transformer embeddings (BERT).
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import joblib


class TFIDFFeatureExtractor:
    """TF-IDF feature extraction."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range (1, 2) = unigrams + bigrams
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.fitted = False
    
    def fit(self, texts):
        """
        Fit TF-IDF vectorizer on training texts.
        
        Args:
            texts: List of text strings
        """
        self.vectorizer.fit(texts)
        self.fitted = True
    
    def transform(self, texts):
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: List of text strings
            
        Returns:
            Sparse matrix of shape (n_texts, n_features)
        """
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """Get feature names (vocabulary)."""
        return self.vectorizer.get_feature_names_out()
    
    def save(self, path):
        """Save vectorizer to disk."""
        joblib.dump(self.vectorizer, path)
    
    def load(self, path):
        """Load vectorizer from disk."""
        self.vectorizer = joblib.load(path)
        self.fitted = True


class WordEmbeddingExtractor:
    """Word embedding feature extraction (Word2Vec, GloVe, FastText)."""
    
    def __init__(self, embedding_model='glove-wiki-300', aggregation='mean'):
        """
        Initialize word embedding extractor.
        
        Args:
            embedding_model: Embedding model name (e.g., 'glove-wiki-300', 'word2vec-google-news-300')
            aggregation: Aggregation method ('mean', 'max', or 'sum')
        """
        self.embedding_model_name = embedding_model
        self.aggregation = aggregation
        print(f"Loading {embedding_model}...")
        self.model = api.load(embedding_model)
        self.embedding_dim = self.model.vector_size
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def _aggregate_embeddings(self, embeddings):
        """
        Aggregate word embeddings to text embedding.
        
        Args:
            embeddings: Array of shape (n_words, embedding_dim)
            
        Returns:
            Array of shape (embedding_dim,)
        """
        if len(embeddings) == 0:
            return np.zeros(self.embedding_dim)
        
        if self.aggregation == 'mean':
            return np.mean(embeddings, axis=0)
        elif self.aggregation == 'max':
            return np.max(embeddings, axis=0)
        elif self.aggregation == 'sum':
            return np.sum(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def extract_features(self, texts):
        """
        Extract word embedding features for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        features = []
        
        for text in tqdm(texts, desc="Extracting word embeddings"):
            words = text.split()
            embeddings = []
            
            for word in words:
                if word in self.model:
                    embeddings.append(self.model[word])
            
            if embeddings:
                text_embedding = self._aggregate_embeddings(np.array(embeddings))
            else:
                text_embedding = np.zeros(self.embedding_dim)
            
            features.append(text_embedding)
        
        return np.array(features)


class TransformerEmbeddingExtractor:
    """Transformer-based feature extraction (BERT, RoBERTa, etc.)."""
    
    def __init__(self, model_name='bert-base-uncased', device=None):
        """
        Initialize transformer embedding extractor.
        
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased', 'roberta-base')
            device: Torch device ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Get embedding dimension
        config = self.model.config
        self.embedding_dim = config.hidden_size
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def extract_features(self, texts, batch_size=32, max_length=512):
        """
        Extract transformer embeddings for texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            max_length: Maximum token sequence length
            
        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting transformer embeddings"):
                batch = texts[i:i + batch_size]
                
                # Tokenize and encode
                encoded = self.tokenizer(
                    batch,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**encoded)
                
                # Extract [CLS] token embeddings
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                features.extend(cls_embeddings)
        
        return np.array(features)


class CombinedFeatureExtractor:
    """Combine multiple feature extraction methods."""
    
    def __init__(self, extractors_dict):
        """
        Initialize combined extractor.
        
        Args:
            extractors_dict: Dict of {feature_name: extractor_instance}
        """
        self.extractors = extractors_dict
    
    def extract_features(self, texts):
        """
        Extract all features.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dict of {feature_name: feature_array}
        """
        features = {}
        
        for name, extractor in self.extractors.items():
            print(f"\nExtracting {name} features...")
            features[name] = extractor.extract_features(texts)
        
        return features
