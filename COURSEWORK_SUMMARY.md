# COMP34812 NLU Coursework — Group Summary & Action Plan

> **Module:** COMP34812 Natural Language Understanding (AY 2025-26)
> **Worth:** 40 marks total
> **Final Deadline:** 3 April 18:00 (all deliverables)

---

## What Is This?

A **shared task** on **pairwise sequence classification**. Pick **one** of three tracks, build **two** different solutions from **two different approach categories**, and submit predictions + code + model cards + poster.

> [!CAUTION]
> **Closed mode** — you may **NOT** use any external datasets. Only the provided training/validation data.

---

## The Three Tracks (Pick One)

| Track | Task | Input Pair | Labels | Train Size | Val Size |
|-------|------|-----------|--------|-----------|---------|
| **A — NLI** | Natural Language Inference | premise, hypothesis | 0/1 (hypothesis true given premise?) | ~24K | ~6K |
| **B — ED** | Evidence Detection | Claim, Evidence | 0/1 (evidence relevant to claim?) | ~21K | ~6K |
| **C — AV** | Authorship Verification | text_1, text_2 | 0/1 (same author?) | ~27K | ~6K |

> **Decision needed:** Agree on which track to do. Registration deadline was **5 March 18:00** via Microsoft Form.

---

## What We Must Build

### Two Solutions from Two Different Categories

| Category | Description | Examples |
|----------|-------------|---------|
| **A** | Unsupervised / traditional ML | TF-IDF + SVM, Bag-of-Words + Logistic Regression, stylometric features |
| **B** | Deep learning **without** transformers | LSTMs, CNNs, GRUs, Siamese networks |
| **C** | Deep learning **with** transformers | Fine-tuned BERT, RoBERTa, DeBERTa, etc. |

> [!IMPORTANT]
> - The two solutions **must** be from **two different categories** (e.g., A+C or B+C).
> - Simple variations within the same category (swapping BERT variants, minor hyperparameter changes) do **not** count.
> - Beating the baseline is not required to pass, but **significantly outperforming** the baseline gets top marks.
> - **Creativity matters** — going beyond standard approaches scores higher.

---

## Deliverables Checklist

### Deliverable 1 — Predictions (CSV)
- [ ] `Group_n_A.csv` / `Group_n_B.csv` / `Group_n_C.csv` (depending on categories used)
- Format: single column `prediction` with 0/1 values, one per test example
- Run each solution on the **test data** (released 1 April 18:00) and generate predictions
- Submit via Canvas by **3 April 18:00**

### Deliverable 2 — Code
- [ ] **Training code** — clearly separated per solution
- [ ] **Evaluation code** — evaluation on the dev set, separated from training
- [ ] **Demo code** — Python notebook per solution that:
  - Installs required packages
  - Loads saved model
  - Takes input test file → outputs predictions CSV
- [ ] **README** with:
  - Code structure explanation
  - How to run
  - Attribution for any reused code/data
  - Links to cloud-stored models (if >10MB, use OneDrive — do NOT put in Canvas)
  - "Use of Generative AI Tools" section if applicable
- [ ] Well-documented, in-line comments throughout

### Deliverable 3 — Model Cards
- [ ] One model card per solution (markdown format)
- [ ] Use the provided template / notebook to generate them
- [ ] Must accurately represent the implemented solutions

### Deliverable 4 — Poster & Flash Presentation (Workshop 4)
- [ ] Single-page PDF poster (A1 landscape or 16:9 PowerPoint slide)
- [ ] Content must include:
  - Task/track description
  - Dataset summary
  - Both methods described
  - Dev results table
  - Brief error analysis
  - Limitations & ethical considerations
- [ ] Flash presentation: 3-4 min poster talk + 3-4 min live demo + 3-4 min Q&A
- [ ] **Both members must attend in person**

### Packaging
- [ ] Compress **everything** (predictions, code, model cards) into **one zip file**
- [ ] Upload to Canvas submission site

---

## Marking Breakdown (40 marks)

| Area | Criterion | Marks |
|------|-----------|-------|
| **Predictions** | Competitive performance — Solution 1 | 3 |
| | Competitive performance — Solution 2 | 3 |
| **Implementation** | Organisation & documentation | 3 |
| | Completeness & reproducibility | 3 |
| | Soundness — Solution 1 | 3 |
| | Soundness — Solution 2 | 3 |
| | Creativity — Solution 1 | 3 |
| | Creativity — Solution 2 | 3 |
| | Evaluation effort | 3 |
| **Model Cards** | Correct formatting | 3 |
| | Informativeness — Solution 1 | 3 |
| | Informativeness — Solution 2 | 3 |
| | Accurate representation | 4 |
| | **Total** | **40** |

---

## Key Dates

| Date | Event |
|------|-------|
| 26 Feb | Trial data released ✅ |
| 5 Mar 18:00 | **Group registration deadline** |
| 8 Mar | Codabench platform launched |
| 12 Mar | Workshop 2 — Codabench overview |
| 19 Mar | Workshop 3 — Deliverable prep advice |
| **1 Apr 18:00** | **Test data released** |
| **3 Apr 18:00** | **ALL deliverables due** |

> [!WARNING]
> Only **2 days** between test data release (1 Apr) and final deadline (3 Apr). Solutions must be ready to run inference before test data drops.

---

## Suggested Task Split

### Before Test Data (now → 1 April)
1. **Pick track** and register group
2. **Explore data** — load trial + training data, check distributions, class balance
3. **Solution 1** (Category ___):
   - Research approach
   - Implement training pipeline
   - Train and tune on train/dev split
   - Evaluate on dev set
4. **Solution 2** (Category ___):
   - Research approach
   - Implement training pipeline
   - Train and tune on train/dev split
   - Evaluate on dev set
5. **Evaluation** — implement evaluation scripts (F1, accuracy, confusion matrix, error analysis)
6. **Model cards** — fill in template for each solution
7. **Demo notebooks** — write inference-only notebooks for each solution
8. **Poster** — draft poster with methods, dev results, analysis

### After Test Data (1 April → 3 April)
9. Run inference on test data with both solutions
10. Generate prediction CSVs
11. Final check of all deliverables
12. Zip and submit

---

## Project Structure

```
NLU_CW/
├── data/
│   ├── trial_data/          # Trial CSVs for all tracks
│   ├── training_data/       # Train & dev splits per track
│   └── predictions.csv      # Example predictions format
├── notebooks/
│   ├── exploration.ipynb    # EDA & data analysis
│   ├── demo_solution1.ipynb # Demo notebook — Solution 1
│   └── demo_solution2.ipynb # Demo notebook — Solution 2
├── src/
│   ├── solution1/           # Training & model code for Solution 1
│   ├── solution2/           # Training & model code for Solution 2
│   ├── evaluation/          # Shared evaluation scripts
│   └── utils/               # Shared utilities (data loading, preprocessing)
├── models/                  # Saved model weights / artifacts (gitignored if large)
├── model_cards/
│   ├── solution1_card.md    # Model card for Solution 1
│   └── solution2_card.md    # Model card for Solution 2
├── outputs/                 # Generated prediction CSVs
├── poster/                  # Poster PDF
├── spec/                    # Coursework specification PDF
├── COURSEWORK_SUMMARY.md    # This file
├── README.md                # Project README (for submission)
└── .gitignore
```
