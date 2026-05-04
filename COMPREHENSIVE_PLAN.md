# COMPLETE REVISED PLAN: Full Dataset + 3 Feature Methods + 3 Models
## BIM432 NLP Project: AI-Generated Text Detection  
## Date: April 29, 2026

---

## **STRATEGY: Work with FULL Dataset + Smart Optimization**

Instead of sampling, use:
1. **Text Truncation:** Limit to first 256 words per document (vs. avg 400)
   - Reduces computation & memory without losing meaning
   - Keeps linguistic patterns for AI/human distinction
   - Reduces embedding dimensions: 487K × 256 words vs. 487K × 400 words

2. **Feature Extraction Strategy:**
   - **TF-IDF:** Sparse matrix → lightweight, works on full dataset ✅
   - **Word Embeddings (GloVe):** Aggregate to single vector → ~600MB total ✅
   - **DistilBERT:** Batch extraction to disk → process in chunks, no OOM ✅

3. **Model Strategy:**
   - **TF-IDF features:** Logistic Regression, SVM, Random Forest (3 models)
   - **Embeddings:** XGBoost, Neural Network (2 additional models)
   - **Total:** 5 models for comprehensive comparison

---

## **WHY THIS WORKS**

### Text Truncation (Key Optimization)
```python
# Before: 487K texts × avg 400 words = ~195M total words
# After: 487K texts × 256 words = ~124M total words (36% reduction)

# Memory savings:
# - DistilBERT: 487K × 256 tokens ≈ 20B tokens → manageable batching
# - GloVe: 487K × 256 words × 300 dims = ~37GB (but aggregated to 1 vector = 600MB)
# - TF-IDF: Sparse matrix, no change in efficiency
```

### Feature Extraction Methods Feasibility

| Method | Input | Processing | Output | Memory | Time |
|--------|-------|-----------|--------|--------|------|
| **TF-IDF** | Full 487K texts | Vectorizer fit/transform (sparse) | 487K × 5000 sparse | 150MB | 2-3 min |
| **GloVe Embeddings** | Full 487K texts (truncated) | Load model, aggregate words to mean vector | 487K × 300 dense | 600MB | 5-10 min |
| **DistilBERT** | Full 487K texts (truncated to 128 tokens) | Batch extraction (1000 at a time), save to disk | 487K × 768 dense (saved as npz) | Disk: 1.5GB, RAM: 500MB | 30-45 min |

### Models Feasibility

| Model | Input Features | Pros | Time |
|-------|---|---|---|
| **Linear SVM** | TF-IDF (150MB) | O(n) complexity, scales to 340K rows, wrapped for probabilities | 2-5 min |
| **XGBoost** | Embeddings (600MB GloVe) | State-of-the-art, handles dense features | 10-15 min |
| **Neural Network** | DistilBERT (768-dim) | Deep learning, end-to-end | 10-20 min |

---

## **COMPLETE PIPELINE**

### **Phase 1: Load Full Dataset** ⏱️ 2-3 min

```python
import pandas as pd
import numpy as np

# Load entire 487K dataset from Drive
df = pd.read_csv('/content/drive/My Drive/path_to_preprocessed_data.csv')

print(f"Dataset loaded: {len(df):,} texts")
print(f"Columns: {df.columns.tolist()}")
print(f"Class distribution:")
print(df['generated'].value_counts())

# Verify text lengths
df['word_count'] = df['text_preprocessed'].str.split().str.len()
print(f"\nText length statistics:")
print(df['word_count'].describe())
```

---

### **Phase 2: Text Truncation to 256 Words** ⏱️ 1-2 min

```python
def truncate_text(text, max_words=256):
    """Truncate text to max_words to reduce computation."""
    if isinstance(text, str):
        words = text.split()
        return ' '.join(words[:max_words])
    return text

print("Truncating texts to 256 words...")
df['text_truncated'] = df['text_preprocessed'].apply(
    lambda x: truncate_text(x, max_words=256)
)

# Verify truncation
print(f"Truncated text lengths:")
df['truncated_word_count'] = df['text_truncated'].str.split().str.len()
print(df['truncated_word_count'].describe())

print(f"Memory reduction: {(1 - df['truncated_word_count'].mean() / df['word_count'].mean())*100:.1f}%")
```

---

### **Phase 3: Train/Test Split on FULL Dataset** ⏱️ <1 min

```python
from sklearn.model_selection import train_test_split

X = df['text_truncated'].values
y = df['generated'].values

# 70/15/15 stratified split on FULL dataset
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/(1-0.15), random_state=42, stratify=y_temp
)

print(f"Full dataset split:")
print(f"  Train: {len(X_train):>8,} samples ({len(X_train)/len(X)*100:>5.1f}%)")
print(f"  Val:   {len(X_val):>8,} samples ({len(X_val)/len(X)*100:>5.1f}%)")
print(f"  Test:  {len(X_test):>8,} samples ({len(X_test)/len(X)*100:>5.1f}%)")
print(f"  Total: {len(X):>8,} samples")

# Verify class balance
for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    unique, counts = np.unique(y_split, return_counts=True)
    props = counts / len(y_split)
    print(f"{split_name}: {props[0]:.1%} Human, {props[1]:.1%} AI")
```

---

## **FEATURE EXTRACTION: 3 METHODS**

### **Method 1: TF-IDF (Full Dataset)** ⏱️ 2-3 min

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

print("\n" + "="*60)
print("FEATURE EXTRACTION METHOD 1: TF-IDF")
print("="*60)

tfidf = TfidfVectorizer(
    max_features=5000,      # More features = more info
    ngram_range=(1, 2),     # Unigrams + bigrams
    min_df=5,               # Min document frequency
    max_df=0.90,
    dtype=np.float32
)

print(f"\nFitting TF-IDF on {len(X_train):,} training texts...")
X_train_tfidf = tfidf.fit_transform(X_train)

print("Transforming validation and test data...")
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nTF-IDF results:")
print(f"  Train shape: {X_train_tfidf.shape}")
print(f"  Val shape:   {X_val_tfidf.shape}")
print(f"  Test shape:  {X_test_tfidf.shape}")
print(f"  Sparsity: {1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]):.1%}")
print(f"  Memory: {X_train_tfidf.data.nbytes / 1e6:.1f} MB")

# Save for later use
sp.save_npz('/content/drive/My Drive/path_to_save/X_train_tfidf.npz', X_train_tfidf)
sp.save_npz('/content/drive/My Drive/path_to_save/X_test_tfidf.npz', X_test_tfidf)
print("✅ Saved to Drive")
```

---

### **Method 2: Word Embeddings (GloVe)** ⏱️ 5-10 min

```python
print("\n" + "="*60)
print("FEATURE EXTRACTION METHOD 2: WORD EMBEDDINGS (GloVe)")
print("="*60)

from gensim.models import KeyedVectors
import gensim.downloader as api

print("\nLoading GloVe embeddings (300-dimensional)...")
try:
    embeddings_model = api.load('glove-wiki-300')
except:
    print("Downloading GloVe model...")
    embeddings_model = api.load('glove-wiki-300')

EMBEDDING_DIM = embeddings_model.vector_size
print(f"✅ Loaded GloVe model (dimension: {EMBEDDING_DIM})")

def get_word_embedding_vector(text, model, aggregation='mean'):
    """
    Convert text to single embedding vector via aggregation.
    Much more memory-efficient than storing individual word vectors.
    """
    words = text.split()
    embeddings = []
    
    for word in words:
        if word in model:
            embeddings.append(model[word])
    
    if embeddings:
        if aggregation == 'mean':
            return np.mean(embeddings, axis=0)
        elif aggregation == 'max':
            return np.max(embeddings, axis=0)
    
    # Return zero vector if no words found
    return np.zeros(EMBEDDING_DIM)

print(f"\nExtracting embeddings for {len(X_train):,} training texts...")
X_train_embeddings = np.array([
    get_word_embedding_vector(text, embeddings_model, 'mean') 
    for text in tqdm(X_train, desc="Train embeddings")
])

print(f"Extracting embeddings for {len(X_val):,} validation texts...")
X_val_embeddings = np.array([
    get_word_embedding_vector(text, embeddings_model, 'mean') 
    for text in tqdm(X_val, desc="Val embeddings")
])

print(f"Extracting embeddings for {len(X_test):,} test texts...")
X_test_embeddings = np.array([
    get_word_embedding_vector(text, embeddings_model, 'mean') 
    for text in tqdm(X_test, desc="Test embeddings")
])

print(f"\nEmbedding results:")
print(f"  Train shape: {X_train_embeddings.shape}")
print(f"  Val shape:   {X_val_embeddings.shape}")
print(f"  Test shape:  {X_test_embeddings.shape}")
print(f"  Memory: {X_train_embeddings.nbytes / 1e6:.1f} MB")

# Save embeddings
np.save('/content/drive/My Drive/path_to_save/X_train_glove.npy', X_train_embeddings)
np.save('/content/drive/My Drive/path_to_save/X_test_glove.npy', X_test_embeddings)
print("✅ Saved to Drive")
```

---

### **Method 3: DistilBERT Embeddings (Batch Processing)** ⏱️ 30-45 min

```python
print("\n" + "="*60)
print("FEATURE EXTRACTION METHOD 3: DISTILBERT EMBEDDINGS")
print("="*60)

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

print("Loading DistilBERT model (faster, lighter than BERT)...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
model.eval()
print("✅ Model loaded")

def extract_distilbert_embeddings_batch(texts, tokenizer, model, device, batch_size=256, max_length=128):
    """
    Extract DistilBERT embeddings with batch processing.
    Saves to disk incrementally to avoid OOM.
    """
    embeddings_list = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting", total=total_batches):
            batch_texts = texts[i:min(i+batch_size, len(texts))]
            
            # Tokenize batch
            encoded = tokenizer(
                list(batch_texts),
                max_length=max_length,  # Truncate to 128 tokens
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass
            outputs = model(**encoded)
            
            # Extract CLS token (first token) embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list.extend(cls_embeddings)
    
    return np.array(embeddings_list)

print(f"\n1. Extracting train embeddings ({len(X_train):,} texts)...")
X_train_distilbert = extract_distilbert_embeddings_batch(X_train, tokenizer, model, device)
print(f"   Shape: {X_train_distilbert.shape}")

print(f"2. Extracting validation embeddings ({len(X_val):,} texts)...")
X_val_distilbert = extract_distilbert_embeddings_batch(X_val, tokenizer, model, device)
print(f"   Shape: {X_val_distilbert.shape}")

print(f"3. Extracting test embeddings ({len(X_test):,} texts)...")
X_test_distilbert = extract_distilbert_embeddings_batch(X_test, tokenizer, model, device)
print(f"   Shape: {X_test_distilbert.shape}")

print(f"\nEmbedding results:")
print(f"  Train memory: {X_train_distilbert.nbytes / 1e9:.2f} GB")
print(f"  Total memory: {(X_train_distilbert.nbytes + X_test_distilbert.nbytes) / 1e9:.2f} GB")

# Save embeddings
np.save('/content/drive/My Drive/path_to_save/X_train_distilbert.npy', X_train_distilbert)
np.save('/content/drive/My Drive/path_to_save/X_val_distilbert.npy', X_val_distilbert)
np.save('/content/drive/My Drive/path_to_save/X_test_distilbert.npy', X_test_distilbert)
print("✅ Saved to Drive")
```

---

## **MODEL TRAINING: 3 MODELS**

### **Models 1-3: TF-IDF Feature Set**

#### **Model 1: Linear Support Vector Machine** ⏱️ 2-5 min

**Why Linear SVM instead of RBF?**
- RBF SVM has O(n²-n³) complexity → would take days on 340K samples and timeout in Colab
- Linear SVM has O(n) complexity → scales efficiently to 340K+ samples  
- Wrapped in `CalibratedClassifierCV` to provide probability estimates for ROC curves
- Still produces high-quality predictions and is widely used in production systems

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

print("\n" + "="*60)
print("MODEL 1: LINEAR SUPPORT VECTOR MACHINE (TF-IDF)")
print("="*60)

# Wrap LinearSVC in CalibratedClassifierCV to get probabilities for ROC curves
base_svm = LinearSVC(C=1.0, random_state=SEED, max_iter=2000, verbose=0)
svm_model = CalibratedClassifierCV(base_svm)

print("Training Linear SVM (this will take 2-5 minutes)...")
svm_model.fit(X_train_tfidf_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_tfidf_scaled)
y_proba_svm = svm_model.predict_proba(X_test_tfidf_scaled)[:, 1]

acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm)
rec_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_proba_svm)

print(f"\nResults:")
print(f"  Accuracy:  {acc_svm:.4f}")
print(f"  Precision: {prec_svm:.4f}")
print(f"  Recall:    {rec_svm:.4f}")
print(f"  F1-Score:  {f1_svm:.4f}")
print(f"  AUC-ROC:   {auc_svm:.4f}")
```

---

### **Model 4: XGBoost (GloVe Embeddings)** ⏱️ 10-15 min

```python
import xgboost as xgb

print("\n" + "="*60)
print("MODEL 4: XGBOOST (GLOVE EMBEDDINGS)")
print("="*60)

# Scale embeddings
scaler_glove = StandardScaler()
X_train_glove_scaled = scaler_glove.fit_transform(X_train_embeddings)
X_test_glove_scaled = scaler_glove.transform(X_test_embeddings)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
xgb_model.fit(X_train_glove_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_glove_scaled)
y_proba_xgb = xgb_model.predict_proba(X_test_glove_scaled)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_xgb):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba_xgb):.4f}")
```

---

### **Model 5: Neural Network (DistilBERT Embeddings)** ⏱️ 20-30 min

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("\n" + "="*60)
print("MODEL 5: NEURAL NETWORK (DISTILBERT EMBEDDINGS)")
print("="*60)

# Prepare PyTorch datasets
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_distilbert),
    torch.LongTensor(y_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test_distilbert),
    torch.LongTensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# Define simple MLP
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_nn = SimpleClassifier().to(device)
optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
print("Training neural network...")
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model_nn(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model_nn.eval()
with torch.no_grad():
    predictions = []
    probabilities = []
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        outputs = model_nn(batch_x)
        probs = torch.softmax(outputs, dim=1)
        predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        probabilities.extend(probs[:, 1].cpu().numpy())

y_pred_nn = np.array(predictions)
y_proba_nn = np.array(probabilities)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_nn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nn):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_nn):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_nn):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_proba_nn):.4f}")
```

---

## **COMPREHENSIVE EVALUATION & COMPARISON**

```python
print("\n" + "="*60)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': [
        'Support Vector Machine',
        'XGBoost',
        'Neural Network'
    ],
    'Feature Set': ['TF-IDF', 'TF-IDF', 'TF-IDF', 'GloVe Embeddings', 'DistilBERT'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_nn)
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_svm),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb),
        precision_score(y_test, y_pred_nn)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_svm),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb),
        recall_score(y_test, y_pred_nn)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_svm),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb),
        f1_score(y_test, y_pred_nn)
    ],
    'AUC-ROC': [
        roc_auc_score(y_test, y_proba_lr),
        roc_auc_score(y_test, y_proba_svm),
        roc_auc_score(y_test, y_proba_rf),
        roc_auc_score(y_test, y_proba_xgb),
        roc_auc_score(y_test, y_proba_nn)
    ]
})

print(f"\n{comparison_df.to_string(index=False)}")

# Best overall model
best_idx = comparison_df['F1-Score'].idxmax()
print(f"\n🏆 Best Overall Model (F1-Score): {comparison_df.loc[best_idx, 'Model']}")
print(f"   ({comparison_df.loc[best_idx, 'F1-Score']:.4f})")

# Best by feature method
for feature_set in comparison_df['Feature Set'].unique():
    subset = comparison_df[comparison_df['Feature Set'] == feature_set]
    best_in_set = subset.loc[subset['F1-Score'].idxmax()]
    print(f"\n🥇 Best in {feature_set}:")
    print(f"   {best_in_set['Model']} (F1: {best_in_set['F1-Score']:.4f})")
```

---

## **VISUALIZATIONS & ERROR ANALYSIS**

### Confusion Matrices (All Models)
```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

model_results = [
    ('Logistic Regression', y_pred_lr),
    ('SVM', y_pred_svm),
    ('Random Forest', y_pred_rf),
    ('XGBoost', y_pred_xgb),
    ('Neural Network', y_pred_nn)
]

for idx, (name, y_pred) in enumerate(model_results):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'],
                ax=axes[idx], cbar=False)
    axes[idx].set_title(f'{name}', fontweight='bold')
    axes[idx].set_ylabel('True')
    axes[idx].set_xlabel('Predicted')

# Hide last subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('all_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
```

### ROC Curves Comparison
```python
fig, ax = plt.subplots(figsize=(10, 8))

model_names = ['LogReg', 'SVM', 'RandomForest', 'XGBoost', 'NeuralNet']
probabilities = [y_proba_lr, y_proba_svm, y_proba_rf, y_proba_xgb, y_proba_nn]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for name, y_proba, color in zip(model_names, probabilities, colors):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})', color=color)

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Feature Importance
```python
# From Logistic Regression
feature_names = tfidf.get_feature_names_out()
coefficients = lr_model.coef_[0]

top_n = 20
top_ai = feature_names[np.argsort(coefficients)[-top_n:][::-1]]
top_human = feature_names[np.argsort(coefficients)[:top_n]]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# AI features
top_ai_coefs = coefficients[np.argsort(coefficients)[-top_n:][::-1]]
axes[0].barh(range(top_n), top_ai_coefs, color='#FF6B6B')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_ai)
axes[0].set_xlabel('Coefficient', fontsize=11)
axes[0].set_title('Top 20 Features → AI-Generated', fontweight='bold', fontsize=12)
axes[0].invert_yaxis()

# Human features
top_human_coefs = coefficients[np.argsort(coefficients)[:top_n]]
axes[1].barh(range(top_n), top_human_coefs[::-1], color='#4ECDC4')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_human[::-1])
axes[1].set_xlabel('Coefficient', fontsize=11)
axes[1].set_title('Top 20 Features → Human-Written', fontweight='bold', fontsize=12)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## **FINAL SUMMARY**

```python
print("\n" + "="*70)
print(" "*20 + "FINAL PROJECT SUMMARY")
print("="*70)

print(f"\n📊 DATASET:")
print(f"  Size: 487,000 texts (FULL dataset)")
print(f"  Text truncation: 256 words per document")
print(f"  Memory reduction: ~36% vs. original")

print(f"\n🔧 FEATURE EXTRACTION (3 Methods):")
print(f"  1. TF-IDF: 5000 features, sparse, 150MB")
print(f"  2. GloVe: 300-dim embeddings, 600MB")
print(f"  3. DistilBERT: 768-dim embeddings, batch-processed, 1.5GB (disk)")

print(f"\n🤖 MODELS (5 Total):")
print(f"  TF-IDF: Logistic Regression, SVM, Random Forest")
print(f"  GloVe: XGBoost")
print(f"  DistilBERT: Neural Network")

print(f"\n📈 RESULTS:")
for idx, row in comparison_df.iterrows():
    print(f"  {row['Model']:25s} | Acc: {row['Accuracy']:.4f} | F1: {row['F1-Score']:.4f} | AUC: {row['AUC-ROC']:.4f}")

print(f"\n✅ DELIVERABLES:")
print(f"  - 3 trained models with metrics")
    print(f"  - Confusion matrices (3 models)")
print(f"  - ROC curves comparison")
print(f"  - Feature importance analysis")
print(f"  - Error analysis and insights")

print(f"\n⏱️ ESTIMATED TOTAL RUNTIME:")
print(f"  Setup & loading: 3 min")
print(f"  Text truncation: 2 min")
print(f"  TF-IDF extraction: 3 min")
print(f"  GloVe extraction: 8 min")
print(f"  DistilBERT extraction: 40 min")
print(f"  Model training (all 5): 60 min")
print(f"  Evaluation & visualization: 10 min")
print(f"  ─────────────────────────")
print(f"  TOTAL: ~2-2.5 hours (feasible in single Colab session)")

print("\n" + "="*70)
```

---

## **KEY ADVANTAGES OF THIS APPROACH**

✅ **Full Dataset Utilized:** 487,000 texts processed end-to-end  
✅ **Smart Optimization:** Text truncation reduces memory 36% without losing meaning  
✅ **Multiple Perspectives:** 3 different feature extraction methods  
✅ **Model Variety:** 3 diverse models across different paradigms
✅ **Feature-Model Synergies:** Each model uses most suitable feature representation  
✅ **Comprehensive Analysis:** Confusion matrices, ROC, feature importance, error analysis  
✅ **Colab Feasible:** 2-2.5 hours runtime fits in free tier  
✅ **Academically Strong:** Demonstrates ML engineering expertise  

---

## **For Your Report**

**Methodology Section:**
"To efficiently process the full 487,000-sample dataset within computational constraints, text truncation to 256 words was applied (maintaining linguistic patterns while reducing computation 36%). Three complementary feature extraction methods were employed: TF-IDF for interpretability, GloVe embeddings for semantic information, and DistilBERT for contextual representation. Three high-impact models spanning traditional ML, gradient boosting, and deep learning were trained for comprehensive evaluation."

**Results Section:**
- Insert comparison table
- Show confusion matrices (3 models)
- Include ROC curves comparison
- Present feature importance findings

**Discussion Section:**
- Compare feature extraction methods
- Analyze model performance trade-offs
- Discuss computational efficiency
- Highlight key discriminative features

---

*Ready to implement. Update Colab notebook paths and run!*
