# REPORT WRITING GUIDE: What's Done vs. What You Need to Fill

## File Location
`c:\Users\HomePC\NLP_project\report\report.tex`

## Compile Instructions
1. Use **Overleaf** (easiest, no installation):
   - Go to overleaf.com
   - Create new project → Upload PDF → Select report.tex
   - Compile with pdfLaTeX

2. Or use **TeXShop**, **MiKTeX**, or **TeX Live** locally

---

## ✅ COMPLETED SECTIONS (Heavy Lifting Done)

### 1. **Abstract** ✅
- Full professional abstract (150+ words)
- Summarizes entire project, results, and contributions

### 2. **Introduction** (85% done)
- ✅ Motivation and problem statement
- ✅ Research objectives
- ✅ Contributions
- ⚠️ **TODO**: Add 2-3 sentences on YOUR personal motivation (Why this topic matters to you?)

### 3. **Related Work** (75% done)
- ✅ Subsection structure on AI-detection, feature extraction, and models
- ✅ Citations and explanations of TF-IDF, GloVe, DistilBERT, SVM, XGBoost
- ⚠️ **TODO**: Add 2-3 paragraphs with proper academic citations on:
  - Existing AI-detection work (OpenAI's work, academic papers)
  - Statistical vs. ML approaches
  - Prior detection accuracy benchmarks

### 4. **Dataset Description** ✅
- ✅ Complete with actual numbers from your notebook
- ✅ 487,235 texts, 62.8% human / 37.2% AI
- ✅ Text length statistics (mean, std dev, min, max)
- ✅ Memory calculations

### 5. **Methodology** ✅ (COMPREHENSIVE)
- ✅ All three feature extraction methods fully explained with math
- ✅ All three models with code snippets
- ✅ Hardware constraints and memory optimization techniques
- ✅ 256-word truncation justification

### 6. **Experiments** ✅
- ✅ Experimental configuration table
- ✅ Complete computational timeline (1.5 hours total)

### 7. **Results** ✅
- ✅ Performance table (Accuracy 99.75%, F1 99.66%, AUC 99.99%)
- ✅ Confusion matrices with error counts
- ✅ Feature importance analysis (top-20 terms)
- ✅ ROC curves description

### 8. **Discussion** (60% done)
- ✅ Model selection trade-offs section
- ✅ Computational cost analysis
- ✅ Data leakage verification
- ⚠️ **TODO SECTIONS**:

  1. **"Why Does Text Truncation Work?"** (2-3 paragraphs)
     - Linguistic justification: Why first 256 words matter
     - Compare to Zipfian distribution of information
     - Cite any evidence about discriminative early text markers
  
  2. **"Interpretability vs. Expressiveness"** (1 paragraph)
     - Your thoughts on trade-offs between simple/interpretable vs. complex/black-box
     - Which is better for real-world deployment and why?
  
  3. **"Domain Specificity"** (2-3 paragraphs)
     - What domains/sources are in your dataset?
     - Would this generalize to other writing types?
     - Limitations to acknowledge
  
  4. **"Why is Accuracy So High?"** (Reflection)
     - Is 99.75% realistic?
     - What would break this model?
     - Adversarial robustness concerns

### 9. **Conclusion** (70% done)
- ✅ Summary of findings
- ✅ Practical implications
- ✅ Future work suggestions structure
- ⚠️ **TODO**: Add 3-5 bullet points under "Future Work" with YOUR ideas
- ⚠️ **TODO**: Write 3-4 sentences of personal reflection in "Concluding Remarks"

### 10. **References** ✅
- ✅ 7 key academic citations provided
- ⚠️ **TODO**: Add more citations as you reference papers in your writing

---

## 📝 SPECIFIC TODO ITEMS (IN ORDER OF PRIORITY)

### Priority 1: Fill Core Missing Content
1. **Introduction - Personal Motivation** (1 paragraph, ~100 words)
   - Why did YOU choose this topic?
   - What inspired the work?

2. **Related Work - Academic Context** (3 paragraphs, ~400 words)
   - Cite 3-5 key papers on AI-detection
   - Explain what gap this work fills
   - Position your work relative to prior art

### Priority 2: Add Interpretive Analysis
3. **Discussion - Text Truncation Justification** (2-3 paragraphs, ~250 words)
   - Defend the 256-word choice linguistically
   - Reference information theory (Zipf's law)

4. **Discussion - Domain Limitations** (2-3 paragraphs, ~250 words)
   - Discuss your dataset's characteristics
   - What domains might be different?

5. **Discussion - Surprising Results** (1-2 paragraphs, ~150 words)
   - Is 99.75% accuracy reasonable?
   - When might this fail?

### Priority 3: Add Future Vision
6. **Conclusion - Future Work Bullets** (5 concrete bullet points)
   - What would you do next?
   - What remains unsolved?

7. **Conclusion - Personal Reflection** (3-4 sentences, ~100 words)
   - What did you learn?
   - Why does this matter?

---

## 🎨 FORMATTING TIPS

- **Figures**: Add `\includegraphics{path/to/image.png}` to reference your PNG files:
  - `NLP_PRJ_Confusion_Matrix.png` → Use in Results
  - `NLP_PRJ_ROC_chart.png` → Use in Results
  - `NLP_PRJ_TF-IDF_word_weight.png` → Use in Feature Importance

- **Figure placement in LaTeX**:
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{../results/figures/NLP_PRJ_Confusion_Matrix.png}
\caption{Confusion matrices for all three models on test set}
\label{fig:confusion}
\end{figure}
```

- **Page count**: Current draft is ~6-7 pages without figures. With your additions + figures, expect 8-10 pages total.

---

## 📊 KEY NUMBERS TO REFERENCE

When writing, use these exact values from your results:

| Metric | Value |
|--------|-------|
| Best Model Accuracy | 99.75% |
| Best F1-Score | 0.9966 |
| Best AUC-ROC | 0.9999 |
| Total Texts | 487,235 |
| Training Time | ~1.5 hours |
| Training Set Size | 341,063 |
| Test Set Size | 73,086 |
| Top AI Indicator | "additionally" |
| Memory Reduction | 36% (via truncation) |

---

## 💡 WRITING PROMPTS FOR YOUR SECTIONS

### For "Related Work - AI Detection":
"Review and summarize 2-3 papers on detecting ChatGPT/GPT-3 generated text. What methods did they use? How did they perform? What's new about YOUR approach?"

### For "Domain Specificity":
"Where did your dataset come from? Academic essays? News articles? Social media? What LLMs were used to generate the AI text (GPT-3, GPT-4, Claude)? How might results differ on poetry, code, or technical writing?"

### For "Text Truncation Justification":
"Why does the first 256 words contain enough information? Discuss: Zipf's Law (important words appear early), structural consistency in LLM output, and any evidence from linguistic research."

### For "Future Work":
"What's the next step? Cross-model evaluation? Adversarial attacks? Fine-tuned BERT? Streaming detection? Multi-class classification (GPT-3 vs. Claude vs. Human)?"

---

## 📋 SUBMISSION CHECKLIST

- [ ] Compile LaTeX successfully (no errors)
- [ ] Add all TODO sections filled in
- [ ] Include figures (Confusion Matrix, ROC, Feature Importance)
- [ ] Verify page count is 8-10 pages
- [ ] Add citations for related work
- [ ] Spell-check and grammar review
- [ ] Export to PDF
- [ ] Submit on Moodle/course platform

---

## 🚀 ESTIMATED TIME TO COMPLETION

- Fill in all TODOs: **2-3 hours**
- Add figures and format: **1 hour**
- Spell-check and refinement: **1 hour**
- **Total: 4-5 hours**

You're doing great! The hard technical part is done—now it's just adding your personal analysis and interpretations.
