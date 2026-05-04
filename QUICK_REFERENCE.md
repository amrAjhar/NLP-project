# Quick Reference Guide

## File Manifest

### Core Python Modules (src/)
| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `preprocessing.py` | Text normalization & tokenization | `TextPreprocessor`, `load_and_preprocess_data()` |
| `feature_extraction.py` | Convert text to numerical features | `TFIDFFeatureExtractor`, `WordEmbeddingExtractor`, `TransformerEmbeddingExtractor` |
| `train_model.py` | Model training & inference | `LogisticRegressionTrainer`, `SVMTrainer`, `BERTTrainer`, `BERTClassifier` |
| `evaluate_model.py` | Evaluation & visualization | `ModelEvaluator`, `ExplainabilityAnalyzer` |
| `utils.py` | Helper utilities | `set_seed()`, `plot_confusion_matrix()`, `print_metrics()` |

### Notebook
| File | Purpose |
|------|---------|
| `notebooks/main.ipynb` | Complete pipeline (EDA → Preprocessing → Feature Extraction → Training → Evaluation → Explainability) |

### Configuration Files
| File | Purpose |
|------|---------|
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore patterns |
| `README.md` | Comprehensive project documentation |

---

## Pipeline Execution Flow

```
1. SETUP & ENVIRONMENT
   ↓
2. LOAD DATA (from Google Drive)
   ↓
3. EDA (statistics, distributions, samples)
   ↓
4. PREPROCESSING (tokenization, stopword removal)
   ↓
5. TRAIN/VAL/TEST SPLIT (stratified, 70/15/15)
   ↓
6. FEATURE EXTRACTION
   ├─ Method A: TF-IDF (5000 features)
   ├─ Method B: Word Embeddings (300 dims)
   └─ Method C: BERT Embeddings (768 dims)
   ↓
7. MODEL TRAINING
   ├─ Model 1: Logistic Regression (TF-IDF)
   ├─ Model 2: SVM (TF-IDF)
   └─ Model 3: BERT Fine-tuning [PRIMARY]
   ↓
8. EVALUATION (metrics, confusion matrices, ROC)
   ↓
9. EXPLAINABILITY (error analysis, feature importance)
   ↓
10. RESULTS SUMMARY & VISUALIZATION
```

---

## Google Colab Checklist

Before running in Colab:
- [ ] Mount Google Drive: `drive.mount('/content/drive')`
- [ ] Update `DATASET_PATH` to your mounted data location
- [ ] Verify GPU availability: Check Colab runtime settings
- [ ] Install dependencies (automated in notebook)
- [ ] Download pre-trained models (automated in notebook)

---

## Key Hyperparameters

### BERT Fine-tuning
```python
learning_rate = 2e-5      # Conservative LR to preserve pre-training
epochs = 3                # Usually converges quickly
batch_size = 16           # Adjust based on GPU memory
warmup_ratio = 0.1        # 10% of steps for warmup
max_seq_length = 512      # BERT's max sequence length
```

### TF-IDF
```python
max_features = 5000       # Vocabulary size
ngram_range = (1, 2)      # Unigrams + bigrams
min_df = 2                # Min document frequency
max_df = 0.95             # Max document frequency
```

### SVM
```python
kernel = 'rbf'            # Non-linear
C = 1.0                   # Regularization parameter
gamma = 'scale'           # Kernel coefficient
```

---

## Expected Output Files

### After Notebook Execution

```
results/
├── figures/
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_bert.png
│   ├── roc_curves_comparison.png
│   ├── metrics_comparison.png
│   ├── error_analysis.png
│   └── feature_importance.png
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression_model.pkl
│   ├── svm_model.pkl
│   └── bert_model.pt
│
└── results_summary.json    # Metrics and stats

```

---

## Performance Benchmarks

**Expected Results (Test Set):**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|----------------|
| Logistic Regression | ~85% | ~84% | ~86% | ~85% | <1 sec |
| SVM | ~87% | ~86% | ~88% | ~87% | ~20 sec |
| BERT | ~92% | ~91% | ~93% | ~92% | ~2-4 hrs |

*Note: Performance varies based on dataset and hyperparameter tuning.*

---

## Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in BERT training
```python
batch_size = 8  # Reduce from 16
```

### Issue: "ModuleNotFoundError: No module named 'transformers'"
**Solution:** Install missing package
```bash
pip install transformers
```

### Issue: "Dataset not found"
**Solution:** Update DATASET_PATH in notebook
```python
DATASET_PATH = '/content/drive/MyDrive/your_folder/data.csv'
```

### Issue: Slow embedding extraction
**Solution:** Use preprocessed texts and smaller batches
```python
X = extract_bert_embeddings(texts, tokenizer, model, device, batch_size=8)
```

---

## Reusing Code in Your Own Project

### Example 1: Just Preprocessing
```python
from src.preprocessing import TextPreprocessor, load_and_preprocess_data

data = load_and_preprocess_data('data.csv')
train_texts = data['train']['texts_preprocessed']
```

### Example 2: Feature Extraction Only
```python
from src.feature_extraction import TFIDFFeatureExtractor

tfidf = TFIDFFeatureExtractor(max_features=3000)
X_train = tfidf.fit_transform(texts)
X_test = tfidf.transform(texts_test)
```

### Example 3: Model Training
```python
from src.train_model import BERTTrainer

trainer = BERTTrainer(device='cuda')
trainer.train(train_texts, train_labels, epochs=3, batch_size=16)
y_pred = trainer.predict(test_texts)
```

### Example 4: Evaluation
```python
from src.evaluate_model import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.compute_metrics(y_test, y_pred, y_pred_proba)
evaluator.plot_confusion_matrix(y_test, y_pred, 'My Model')
```

---

## GitHub Setup

### Initialize Repository
```bash
cd nlp-ai-text-detection
git init
git add .
git commit -m "Initial commit: Complete NLP AI-text detection pipeline"
git remote add origin https://github.com/username/nlp-ai-text-detection.git
git push -u origin main
```

### Create .gitattributes (for large files)
```
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
```

---

## Project Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| EDA & Setup | 1-2 hrs | Data insights, visualizations |
| Preprocessing | 1 hr | Cleaned, split data |
| Feature Extraction | 2-3 hrs | TF-IDF, embeddings, BERT features |
| Model Training | 3-4 hrs | Trained models |
| Evaluation | 1-2 hrs | Metrics, confusion matrices, ROC |
| Explainability | 2-3 hrs | Error analysis, feature importance |
| Report & Documentation | 2-3 hrs | Final 8-10 page report |
| **Total** | **~15-20 hrs** | Complete project |

---

## Next Steps After Colab Execution

1. **Review Results:** Check metrics, confusion matrices, ROC curves
2. **Analyze Errors:** Examine false positives/negatives
3. **Optimize Hyperparameters:** If performance unsatisfactory, tune learning rate, epochs, etc.
4. **Write Report:** Document methodology, results, discussion
5. **Push to GitHub:** Commit code, results, and report
6. **Prepare Presentation:** Create slides for 20-25 minute presentation

---

**For detailed documentation, refer to README.md**
