# AI-Generated Text Detection: Comprehensive NLP Pipeline

**A production-ready machine learning system for detecting AI-generated text using multiple feature extraction methods and classification algorithms.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Academic Project](https://img.shields.io/badge/Academic%20Project-BIM432%20NLP-green.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Evaluation](#-evaluation)
- [Key Findings](#-key-findings)
- [Citation](#-citation)

---

## 🎯 Overview

This project implements a **comprehensive machine learning pipeline** to distinguish between human-authored and AI-generated text. Using **487,235 real-world text samples**, we evaluate three complementary feature extraction methods and three diverse classification models to achieve near-perfect detection accuracy.

**Key Innovation:** Processing the *complete* dataset (no sampling) on resource-constrained hardware (Google Colab, 12.7GB RAM) through careful optimization including text truncation and memory-aware batch processing.

**Use Cases:**
- 🎓 Detect AI-assisted plagiarism in academic settings
- 📰 Identify AI-generated content in news and media
- 💬 Flag AI-generated spam on social platforms
- ✅ Verify content authenticity in publication workflows

---

## 🏆 Key Results

| Model | Features | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|----------|---------|
| **Linear SVM** (Best) | TF-IDF | **99.75%** | **99.66%** | **99.99%** |
| Neural Network | DistilBERT | 99.16% | 98.86% | 99.96% |
| XGBoost | GloVe | 96.91% | 95.79% | 99.57% |

**Best Model:** Linear SVM trained on TF-IDF features
- **Training time:** <1 minute
- **Inference time:** <100ms per text
- **Memory footprint:** ~300MB

---

## 📂 Project Structure

```
project/
│
├── data/
│   └── dataset_sample.csv           # Sample dataset (full dataset: 487,235 texts)
│
├── notebooks/
│   └── exploration.ipynb             # Complete exploration + full pipeline
│
├── src/
│   ├── preprocessing.py              # Text preprocessing & tokenization
│   ├── feature_extraction.py         # TF-IDF, GloVe, DistilBERT extraction
│   ├── train_model.py                # Model training functions
│   ├── evaluate_model.py             # Evaluation & metrics
│   └── utils.py                      # Helper utilities
│
├── results/
│   ├── figures/                      # Generated visualizations
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   └── feature_importance.png
│   └── models/                       # Trained model artifacts
│       ├── svm_model.joblib
│       ├── xgboost_model.json
│       ├── distilbert_nn_weights.pth
│       └── tfidf_vectorizer.joblib
│
├── report/
│   ├── report.tex                    # LaTeX technical report (8+ pages)
│   └── report.pdf                    # Compiled PDF report
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── README.md                         # This file
└── LICENSE                           # MIT License

```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

```python
# 1. Upload notebook/exploration.ipynb to Colab
# 2. Mount Google Drive for large dataset
from google.colab import drive
drive.mount('/content/drive')

# 3. Run all cells in order (~1.5 hours for full dataset)
```

### Option 2: Local Machine

```bash
# 1. Clone repository
git clone https://github.com/yourusername/NLP_project.git
cd NLP_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebook
jupyter notebook notebooks/exploration.ipynb
```

---

## 📦 Installation

### Requirements
- **Python:** 3.8 or higher
- **RAM:** 12GB minimum (for full dataset processing)
- **GPU:** Optional (CUDA recommended for DistilBERT extraction)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/NLP_project.git
cd NLP_project
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate

# Or using conda
conda create -n nlp-detection python=3.8
conda activate nlp-detection
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `pandas` (1.0+) - Data manipulation
- `numpy` (1.19+) - Numerical computing
- `scikit-learn` (0.24+) - ML algorithms
- `torch` (1.9+) - Neural networks
- `transformers` (4.0+) - BERT/DistilBERT
- `xgboost` (1.3+) - Gradient boosting
- `gensim` (3.8+) - Word embeddings (GloVe)
- `matplotlib` (3.1+) - Visualization
- `seaborn` (0.11+) - Statistical plots

---

## 📊 Usage

### 1. Running the Complete Pipeline

```python
# In notebooks/exploration.ipynb

# All sections run in order:
# 1. Environment Setup
# 2. Dataset Exploration & Visualization
# 3. Data Preprocessing & Stratified Splitting
# 4. Feature Extraction (3 methods)
# 5. Model Training & Evaluation
# 6. Results Comparison & Visualization
```

### 2. Using Individual Modules

```python
from src.preprocessing import TextPreprocessor
from src.feature_extraction import TFIDFFeatureExtractor, WordEmbeddingExtractor
from src.train_model import LogisticRegressionTrainer
from src.evaluate_model import ModelEvaluator

# Initialize preprocessor
preprocessor = TextPreprocessor(lowercase=True, remove_stopwords=True)
texts = [preprocessor.preprocess(text) for text in raw_texts]

# Extract features
tfidf_extractor = TFIDFFeatureExtractor(max_features=5000)
X_tfidf = tfidf_extractor.fit_transform(texts)

# Train model
trainer = LogisticRegressionTrainer()
metrics = trainer.train(X_tfidf, y_train, X_val_tfidf, y_val)
```

### 3. Making Predictions

```python
# Load trained model
import joblib
svm_model = joblib.load('results/models/svm_model.joblib')
vectorizer = joblib.load('results/models/tfidf_vectorizer.joblib')

# Preprocess new text
text = "Your text here..."
X = vectorizer.transform([text])

# Predict
prediction = svm_model.predict(X)[0]
probability = svm_model.decision_function(X)[0]

class_name = "AI-Generated" if prediction == 1 else "Human-Written"
print(f"Classification: {class_name}")
print(f"Confidence: {abs(probability):.4f}")
```

---

## 📥 Dataset

### Source
Kaggle AI vs. Human Text Dataset (referenced in the project)

### Format
```csv
text,generated
"This is a human-written sample...",0
"This text was generated by GPT-3...",1
...
```

### Statistics
- **Total samples:** 487,235
- **Human-written:** 305,797 (62.8%)
- **AI-generated:** 181,438 (37.2%)
- **Avg. text length:** 380 words (mean: 421 human, 344 AI)
- **Class balance:** Maintained across train/val/test splits

### Preprocessing
- ✅ Removed null/empty texts
- ✅ Truncated to 256 words per document (memory optimization)
- ✅ Preserved original text for feature extraction
- ✅ Applied stratified splitting (70/15/15)

---

## 🔧 Methodology

### Feature Extraction Methods

#### 1. **TF-IDF** (Term Frequency-Inverse Document Frequency)
- **Why:** Lightweight, interpretable, effective for text classification
- **Config:** 5,000 features (unigrams + bigrams), 96.7% sparse
- **Time:** ~3 minutes for 341K texts
- **Memory:** 226.5 MB

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90
)
```

#### 2. **GloVe Embeddings** (Global Vectors for Word Representation)
- **Why:** Captures semantic relationships, 300-dimensional
- **Config:** Mean-pooled word vectors from pre-trained GloVe model
- **Time:** ~5 minutes for 341K texts
- **Memory:** 818.6 MB

#### 3. **DistilBERT** (Distilled Bidirectional Encoder Representations)
- **Why:** State-of-the-art contextual embeddings, lightweight BERT variant
- **Config:** 768-dimensional embeddings, batch-processed (256 texts/batch)
- **Time:** ~37 minutes for 341K texts (GPU-accelerated)
- **Memory:** 1.50 GB

### Classification Models

#### 1. **Linear SVM** (Support Vector Machine via SGD)
```python
SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000)
```
- **Advantage:** O(1) memory overhead, linear time complexity
- **Why not RBF?** O(n²) complexity would take days on 341K samples
- **Result:** 99.75% accuracy

#### 2. **XGBoost** (Extreme Gradient Boosting)
```python
XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1)
```
- **Advantage:** Non-linear decision boundaries, handles class imbalance
- **Result:** 96.91% accuracy

#### 3. **Neural Network** (3-layer fully-connected)
```
Input (768) → Linear(256) + ReLU + Dropout(0.3)
           → Linear(128) + ReLU + Dropout(0.3)
           → Linear(2) + Softmax → Output
```
- **Advantage:** Deep learning baseline, can learn complex transformations
- **Result:** 99.16% accuracy

---

## 📈 Evaluation

### Metrics
- **Accuracy:** Overall correct predictions
- **Precision:** Of predicted AI texts, how many were correct?
- **Recall:** Of actual AI texts, how many were detected?
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Area under the receiver operating characteristic curve
- **Confusion Matrix:** True/false positives and negatives

### Visualizations
1. **Confusion Matrices:** Error patterns for each model
2. **ROC Curves:** Trade-off between TPR and FPR
3. **Feature Importance:** Top discriminative terms
4. **Class Distribution:** Training set composition

### Results Summary
- **SVM achieves 99.75% accuracy** with minimal training time (15 seconds)
- **Neural Network closely follows (99.16%)** at 20x training time
- **XGBoost lags (96.91%)** despite using semantic GloVe embeddings
- **Conclusion:** Simpler ≠ worse; TF-IDF captures discriminative patterns effectively

---

## 🔍 Key Findings

### Linguistic Patterns (Top AI Indicators)

The SVM feature weights reveal that **formal discourse markers** are strong predictors of AI-generated text:

| Rank | Term | Weight |
|------|------|--------|
| 1 | additionally | 0.0847 |
| 2 | further | 0.0756 |
| 3 | moreover | 0.0712 |
| 4 | overall | 0.0698 |
| 5 | thus | 0.0685 |

**Interpretation:** LLMs are trained with RLHF to produce structured, coherent text with explicit logical connectors, whereas humans write with more stylistic variation.

### Model Performance Trade-offs

| Dimension | Best | Trade-off |
|-----------|------|-----------|
| Accuracy | SVM (99.75%) | NN is competitive (99.16%) |
| Interpretability | SVM/TF-IDF (feature weights visible) | NN is black-box |
| Speed | SVM (15 sec training) | NN is 80x slower |
| Memory | SVM (300MB total) | DistilBERT is 5x larger |

**Recommendation:** For production systems, use **SVM + TF-IDF** for speed, interpretability, and resource efficiency.

---

## 🔐 Limitations & Future Work

### Limitations
- ⚠️ Dataset contains 487K texts but **no adversarially-crafted examples**
- ⚠️ **Single LLM source:** Unknown how well this generalizes to GPT-4, Claude, Gemini
- ⚠️ **Domain-specific:** Trained on general text; may underperform on code, poetry, technical docs
- ⚠️ **Fixed-size inputs:** 256-word truncation may lose context for longer documents

### Future Directions
1. **Cross-LLM Evaluation:** Test on text from different LLM architectures
2. **Adversarial Robustness:** Evaluate against paraphrased/edited AI text
3. **Cross-Domain Transfer:** Benchmark on different writing domains
4. **Fine-tuned Transformers:** End-to-end BERT training (vs. feature extraction)
5. **Ensemble Methods:** Combine all three features for potential boost
6. **Real-time Deployment:** Optimize for <100ms inference in production

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{nlp_ai_detection_2026,
  title={AI-Generated Text Detection: Comprehensive NLP Pipeline},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/NLP_project}}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Student Name**  
BIM432 Natural Language Processing Course  
Academic Year 2026

---

## 🙏 Acknowledgments

- Dataset: Kaggle AI vs. Human Text
- Course Instructor: [BIM432 Instructors]
- Libraries: scikit-learn, PyTorch, Hugging Face Transformers, XGBoost

---

## 📧 Questions?

For questions or issues, please open a GitHub issue or contact the project author.

---

**Last Updated:** May 2026  
**Status:** ✅ Complete and production-ready

│
├── data/
│   └── dataset_sample.csv          # Sample/reference data
│
├── notebooks/
│   └── main.ipynb                  # Main Colab notebook (complete pipeline)
│
├── src/
│   ├── preprocessing.py            # Text preprocessing utilities
│   ├── feature_extraction.py       # Feature extraction methods
│   ├── train_model.py              # Model training classes
│   ├── evaluate_model.py           # Evaluation & visualization
│   └── utils.py                    # Helper functions
│
├── results/
│   ├── figures/                    # Generated visualizations
│   │   ├── confusion_matrix_*.png
│   │   ├── roc_curves_comparison.png
│   │   └── metrics_comparison.png
│   └── models/                     # Trained model artifacts
│       ├── tfidf_vectorizer.pkl
│       ├── logistic_regression.pkl
│       ├── svm_model.pkl
│       └── bert_model.pt
│
├── report/
│   └── project_report.pdf          # Final technical report
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore patterns
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or Local environment with GPU

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/nlp-ai-text-detection.git
cd nlp-ai-text-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Dataset
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
2. Upload to Google Drive or local `data/` folder
3. Update dataset path in notebook

### Step 4: Run the Pipeline
**Option A: Google Colab (Recommended)**
1. Upload notebook to Google Colab
2. Mount Google Drive: `drive.mount('/content/drive')`
3. Update `DATASET_PATH` to your mounted data
4. Run all cells sequentially

**Option B: Local Jupyter**
```bash
jupyter notebook notebooks/main.ipynb
```

## Usage

### Quick Start in Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load and run the notebook
!git clone https://github.com/your-username/nlp-ai-text-detection.git
%cd nlp-ai-text-detection
```

### Using Individual Modules

```python
from src.preprocessing import TextPreprocessor, load_and_preprocess_data
from src.feature_extraction import TFIDFFeatureExtractor, TransformerEmbeddingExtractor
from src.train_model import LogisticRegressionTrainer, BERTTrainer
from src.evaluate_model import ModelEvaluator

# Load and preprocess data
data = load_and_preprocess_data('data/dataset.csv')

# Extract features
tfidf = TFIDFFeatureExtractor()
X_train_tfidf = tfidf.fit_transform(data['train']['texts_preprocessed'])

# Train model
trainer = LogisticRegressionTrainer()
trainer.train(X_train_tfidf, data['train']['labels'])

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.compute_metrics(y_test, y_pred)
```

## Methodology

### 1. Data Exploration (EDA)
- Class distribution analysis
- Text length statistics (characters, words, tokens)
- Vocabulary analysis (top N words per class)
- Sample text inspection

**Output:** Histograms, word clouds, descriptive statistics

### 2. Preprocessing
- **Tokenization:** Split text into words/tokens
- **Lowercasing:** Normalize case
- **Punctuation Removal:** Remove punctuation marks
- **Stopword Removal:** Filter common words (optional)
- **Lemmatization:** Reduce words to base forms (optional)

**Justification:** Preprocessing reduces noise and standardizes input, improving model training.

### 3. Feature Extraction

#### Method A: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Formula:** TF-IDF = TF × IDF
- **Dimensionality:** 5000 features (configurable)
- **Interpretation:** Each feature represents word importance in individual documents
- **Pros:** Fast, interpretable, sparse representation
- **Cons:** Loses word order and semantic relationships

#### Method B: Word Embeddings (GloVe)
- **Pre-trained Model:** GloVe (Global Vectors, 300-dim)
- **Process:** Tokenize → Lookup embeddings → Aggregate (mean pooling)
- **Dimensionality:** 300 features
- **Interpretation:** Each feature represents semantic dimensions
- **Pros:** Captures semantic relationships (similar words have similar vectors)
- **Cons:** Static embeddings (context-independent)

#### Method C: Transformer Embeddings (BERT)
- **Model:** BERT-base-uncased (110M parameters)
- **Process:** Tokenize → Forward pass → Extract [CLS] token
- **Dimensionality:** 768 features
- **Interpretation:** [CLS] token aggregates document semantics
- **Pros:** Context-dependent, state-of-the-art performance
- **Cons:** Computationally expensive, less interpretable

### 4. Model Training

#### Model 1: Logistic Regression + TF-IDF
- **Algorithm:** Linear binary classifier with logistic function
- **Decision Boundary:** Linear in feature space
- **Training:** Fast (<1 second)
- **Interpretability:** High (feature weights directly indicate importance)
- **Purpose:** Baseline model, interpretability benchmark

#### Model 2: Support Vector Machine (SVM) + TF-IDF
- **Kernel:** RBF (Radial Basis Function)
- **Decision Boundary:** Non-linear (maps to higher-dimensional space)
- **Training:** Moderate (~10-30 seconds)
- **Interpretability:** Medium
- **Purpose:** Non-linear alternative to logistic regression

#### Model 3: BERT Fine-tuning [PRIMARY]
- **Architecture:** 
  ```
  Input text → BERT Tokenizer → BERT Transformer (12 layers) 
  → [CLS] token → Dropout → Linear classifier → Softmax → Output
  ```
- **Training:** Transfer learning (update all BERT weights)
- **Hyperparameters:**
  - Learning rate: 2e-5 (small to preserve pre-training)
  - Epochs: 3-5
  - Batch size: 16-32
  - Warmup: 10% of total steps
- **Training Time:** 2-4 hours on GPU
- **Purpose:** Capture complex linguistic patterns learned from 3.3B words

### 5. Model Evaluation

**Metrics:**
- **Accuracy:** (TP + TN) / Total — Overall correctness
- **Precision:** TP / (TP + FP) — False positive rate (human texts incorrectly flagged)
- **Recall:** TP / (TP + FN) — False negative rate (AI texts missed)
- **F1-Score:** 2×(Precision×Recall)/(Precision+Recall) — Harmonic mean
- **AUC-ROC:** Area under ROC curve — Threshold-independent performance

**Visualizations:**
- Confusion matrices (per model)
- ROC curves (overlay comparison)
- Metrics bar charts
- Error distribution plots

### 6. Explainability Analysis

- **Misclassification Analysis:** Categorize false positives (human→AI) and false negatives (AI→human)
- **Feature Importance:** Extract top discriminative features from Logistic Regression
- **Error Patterns:** Analyze text length, vocabulary, and stylistic patterns in misclassifications

## Results

### Expected Performance (Baseline Estimates)
- **Logistic Regression:** 85-88% accuracy
- **SVM:** 87-90% accuracy
- **BERT:** 90-95% accuracy (depending on dataset quality)

### Evaluation Metrics (Test Set)
Results will be displayed in the notebook with confusion matrices, ROC curves, and detailed metrics.

## Technical Specifications

### Hardware Requirements
- **GPU:** Strongly recommended for BERT fine-tuning (50x speedup)
- **Memory:** 8GB+ RAM, 4GB+ VRAM (for batch size 16-32)
- **Storage:** ~500MB for models and data

### Software Stack
- **Language:** Python 3.8+
- **Deep Learning:** PyTorch 2.0+, Transformers 4.30+
- **ML:** scikit-learn 1.3+
- **NLP:** NLTK, spaCy
- **Visualization:** Matplotlib, Seaborn
- **Explainability:** LIME, SHAP

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution:** Reduce batch size or use gradient accumulation
```python
batch_size = 8  # Reduce from 16
gradient_accumulation_steps = 2
```

### Issue: BERT embeddings too slow
**Solution:** Use CPU or reduce dataset size for testing
```python
device = torch.device('cpu')  # Or use AWS/Colab GPU
```

### Issue: Word embeddings won't load
**Solution:** Check gensim installation or skip this feature
```bash
pip install --upgrade gensim
```

## Model Interpretability

### Feature Importance (TF-IDF Models)
- High-weight features indicate discriminative patterns
- Example: "therefore", "implement", "utilize" → AI indicators
- Example: "lol", "like", "honestly" → Human indicators

### BERT Attention Analysis
- Attention weights show token interdependencies
- Early layers: Capture syntax and grammar
- Late layers: Capture semantics and task-specific patterns

## Limitations & Future Work

### Current Limitations
1. **Domain Bias:** Model trained on specific dataset; may not generalize to different AI models or writing styles
2. **Adversarial Robustness:** System vulnerable to paraphrasing or character-level noise
3. **Computational Cost:** BERT fine-tuning requires GPU; TF-IDF is more accessible
4. **Black Box Nature:** BERT decisions are less interpretable than linear models

### Future Enhancements
1. **Cross-domain Evaluation:** Test on multiple AI models (GPT-4, Claude, etc.)
2. **Ensemble Diversity:** Combine multiple feature extraction methods
3. **Adversarial Training:** Improve robustness via adversarial examples
4. **Lightweight Models:** Distill BERT into smaller, faster models
5. **Real-time Deployment:** Convert to ONNX format for edge deployment

## Report Structure

The final project report (8-10 pages) includes:
1. **Introduction:** Problem motivation and importance
2. **Related Work:** Survey of existing detection methods
3. **Dataset Description:** Data source, statistics, preprocessing rationale
4. **Methodology:** Feature extraction and model architectures with technical detail
5. **Experiments:** Training procedures, hyperparameters, hardware specs
6. **Results:** Metrics tables, confusion matrices, ROC curves
7. **Discussion:** Why BERT works, error analysis, limitations
8. **Conclusion:** Takeaways and future directions

## Academic Integrity

This project was developed according to academic integrity standards:
- ✓ Original code and analysis
- ✓ Proper citation of external sources and datasets
- ✓ No code copying between groups
- ✓ All external resources credited

## Citation

If you use this code or methodology in your work, please cite:

```bibtex
@misc{nlp_ai_text_detection_2026,
  title={AI-Generated Text Detection: NLP Pipeline with Transformers},
  author={Your Name and Team},
  year={2026},
  howpublished={GitHub Repository},
  url={https://github.com/your-username/nlp-ai-text-detection}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit a Pull Request

## Contact

For questions or issues, please open a GitHub issue or contact the project maintainers.

---

**Last Updated:** April 2026  
**Status:** ✓ Complete
