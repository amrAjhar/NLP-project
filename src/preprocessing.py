"""
Data preprocessing module for NLP text classification.
Handles tokenization, lowercasing, punctuation removal, stopword removal, and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextPreprocessor:
    """Text preprocessing pipeline."""
    
    def __init__(self, lowercase=True, remove_punctuation=True, 
                 remove_stopwords=True, lemmatize=False):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Whether to lowercase text
            remove_punctuation: Whether to remove punctuation
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """
        Apply preprocessing steps to text.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        # Step 1: Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Step 2: Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Step 3: Remove extra whitespace
        text = ' '.join(text.split())
        
        # Step 4: Tokenize
        tokens = word_tokenize(text)
        
        # Step 5: Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Step 6: Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Rejoin tokens
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


def extract_text_features(text):
    """
    Extract lexical features from text.
    
    Args:
        text: Text string
        
    Returns:
        Dict of features
    """
    tokens = word_tokenize(text.lower())
    
    # Basic statistics
    num_tokens = len(tokens)
    num_unique_tokens = len(set(tokens))
    avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0
    
    # Punctuation ratio (count punctuation in original text)
    num_punctuation = sum(1 for c in text if c in string.punctuation)
    punctuation_ratio = num_punctuation / len(text) if len(text) > 0 else 0
    
    # Digit ratio
    num_digits = sum(1 for c in text if c.isdigit())
    digit_ratio = num_digits / len(text) if len(text) > 0 else 0
    
    # Uppercase ratio
    num_uppercase = sum(1 for c in text if c.isupper())
    uppercase_ratio = num_uppercase / len(text) if len(text) > 0 else 0
    
    return {
        'num_tokens': num_tokens,
        'num_unique_tokens': num_unique_tokens,
        'avg_token_length': avg_token_length,
        'punctuation_ratio': punctuation_ratio,
        'digit_ratio': digit_ratio,
        'uppercase_ratio': uppercase_ratio,
        'type_token_ratio': num_unique_tokens / num_tokens if num_tokens > 0 else 0
    }


def load_and_preprocess_data(data_path, text_column='text', label_column='label',
                             preprocessing_config=None, test_size=0.15, val_size=0.15, 
                             random_state=42):
    """
    Load dataset and apply preprocessing with stratified split.
    
    Args:
        data_path: Path to CSV dataset
        text_column: Column name for text
        label_column: Column name for labels (assumed to be 0=human, 1=AI)
        preprocessing_config: Dict with preprocessing parameters
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation
        random_state: Random seed
        
    Returns:
        Dict with train/val/test splits
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor
    if preprocessing_config is None:
        preprocessing_config = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': False
        }
    
    preprocessor = TextPreprocessor(**preprocessing_config)
    
    # Preprocess texts
    print("Preprocessing texts...")
    df['text_preprocessed'] = df[text_column].apply(preprocessor.preprocess)
    
    # Extract lexical features
    print("Extracting lexical features...")
    df['num_tokens'] = df[text_column].apply(lambda x: len(word_tokenize(x)))
    df['avg_token_length'] = df[text_column].apply(
        lambda x: np.mean([len(t) for t in word_tokenize(x)]) if word_tokenize(x) else 0
    )
    df['punctuation_ratio'] = df[text_column].apply(
        lambda x: sum(1 for c in x if c in string.punctuation) / len(x) if len(x) > 0 else 0
    )
    
    # Stratified split: train -> split into train/val
    X = df[['text', 'text_preprocessed']].values
    y = df[label_column].values
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val (from remaining data)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    # Reconstruct texts and preprocessed texts
    def split_texts(X_split):
        texts = [x[0] for x in X_split]
        texts_preprocessed = [x[1] for x in X_split]
        return texts, texts_preprocessed
    
    train_texts, train_texts_preprocessed = split_texts(X_train)
    val_texts, val_texts_preprocessed = split_texts(X_val)
    test_texts, test_texts_preprocessed = split_texts(X_test)
    
    print(f"\nDataset Split:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Validation: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    print(f"\nClass Distribution (Train):")
    print(f"  Human: {(y_train == 0).sum()} ({100 * (y_train == 0).sum() / len(y_train):.1f}%)")
    print(f"  AI: {(y_train == 1).sum()} ({100 * (y_train == 1).sum() / len(y_train):.1f}%)")
    
    return {
        'train': {
            'texts': train_texts,
            'texts_preprocessed': train_texts_preprocessed,
            'labels': y_train
        },
        'val': {
            'texts': val_texts,
            'texts_preprocessed': val_texts_preprocessed,
            'labels': y_val
        },
        'test': {
            'texts': test_texts,
            'texts_preprocessed': test_texts_preprocessed,
            'labels': y_test
        },
        'preprocessor': preprocessor
    }


import numpy as np
