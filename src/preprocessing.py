"""
Text preprocessing module for AI-Generated Text Detection project.
Handles text truncation, cleaning, and preparation for feature extraction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def truncate_text(text, max_words=256):
    """
    Truncate text to max_words efficiently.
    
    Args:
        text: Input text string
        max_words: Maximum number of words to keep (default: 256)
    
    Returns:
        Truncated text string
    """
    if isinstance(text, str):
        words = text.split(None, max_words)
        return ' '.join(words[:max_words])
    return ""


def load_and_preprocess_dataset(dataset_path, text_column=None, label_column=None, max_words=256):
    """
    Load dataset and perform initial preprocessing.
    
    Args:
        dataset_path: Path to CSV file with texts and labels
        text_column: Name of text column (auto-detected if None)
        label_column: Name of label column (auto-detected if None)
        max_words: Maximum words per document (default: 256)
    
    Returns:
        df: Preprocessed DataFrame
        TEXT_COLUMN: Name of text column
        LABEL_COLUMN: Name of label column
    """
    print(f"Loading dataset from: {dataset_path}\n")
    
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded: {len(df):,} texts")
    print(f"\nDataset info:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df):,}")
    print(f"  Data types:\n{df.dtypes}")
    
    # Auto-detect columns if not provided
    if text_column is None:
        for col in ['text', 'text_preprocessed', 'content']:
            if col in df.columns:
                text_column = col
                break
    
    if label_column is None:
        for col in ['generated', 'label', 'target']:
            if col in df.columns:
                label_column = col
                break
    
    if text_column is None or label_column is None:
        raise ValueError("Could not auto-detect TEXT_COLUMN and LABEL_COLUMN")
    
    print(f"\n✅ Auto-detected columns:")
    print(f"  TEXT_COLUMN: '{text_column}'")
    print(f"  LABEL_COLUMN: '{label_column}'")
    
    # Truncate texts
    print(f"\n{'='*60}")
    print("STEP 1: MEMORY-OPTIMIZED TEXT TRUNCATION")
    print(f"{'='*60}\n")
    
    print(f"Truncating {len(df):,} rows to {max_words} words...")
    df['text_truncated'] = df[text_column].apply(lambda x: truncate_text(x, max_words))
    
    print("Dropping original text column to recover RAM...")
    df.drop(columns=[text_column], inplace=True)
    
    import gc
    gc.collect()
    
    print(f"\nCalculating statistics...")
    df['word_count'] = df['text_truncated'].apply(lambda x: x.count(' ') + 1)
    print(df['word_count'].describe())
    
    print(f"\n✅ Truncation complete!")
    
    print(f"\nClass distribution:")
    print(df[label_column].value_counts())
    print(f"Proportions: {(df[label_column].value_counts() / len(df)).round(3).to_dict()}")
    
    return df, 'text_truncated', label_column


def train_val_test_split(X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into train/val/test with stratification.
    
    Args:
        X: Features (text or array)
        y: Labels
        train_size: Proportion for training (default: 0.70)
        val_size: Proportion for validation (default: 0.15)
        test_size: Proportion for testing (default: 0.15)
        random_state: Random seed (default: 42)
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print(f"\n{'='*60}")
    print("STEP 2: TRAIN/VAL/TEST SPLIT (70/15/15)")
    print(f"{'='*60}\n")
    
    # Split 1: 70% train, 30% temp
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split 2: Split temp into train/val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"Full dataset split:")
    print(f"  Train: {len(X_train):>8,} samples ({len(X_train)/(len(X_train)+len(X_val)+len(X_test))*100:>5.1f}%)")
    print(f"  Val:   {len(X_val):>8,} samples ({len(X_val)/(len(X_train)+len(X_val)+len(X_test))*100:>5.1f}%)")
    print(f"  Test:  {len(X_test):>8,} samples ({len(X_test)/(len(X_train)+len(X_val)+len(X_test))*100:>5.1f}%)")
    print(f"  Total: {len(X_train)+len(X_val)+len(X_test):>8,} samples")
    
    print(f"\nClass distribution verification:")
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        props = counts / len(y_split)
        print(f"  {split_name}: {props[0]:>5.1%} Human, {props[1]:>5.1%} AI")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
