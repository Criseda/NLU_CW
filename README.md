# COMP34812 NLU Shared Task

> **Track:** C — Authorship Verification

This repository contains two distinct solutions for the Authorship Verification shared task, focusing on pairwise sequence classification.

## Project Structure

```
NLU_CW/
├── data/                    # All data (trial, training, test)
├── notebooks/
│   ├── demo_solution1.ipynb # Inference demo — Solution 1
│   └── demo_solution2.ipynb # Inference demo — Solution 2
├── src/
│   ├── solution1/           # Stylometric Extraction + Stacking Ensemble
│   ├── solution2/           # Transformer Cross-Encoders + Neural Meta-Learner
│   └── evaluate.py          # Unified evaluation script (Metrics calculation)
├── models/                  # Saved model weights
├── model_cards/             # Model cards detailing architecture and performance
│   ├── solution1_card.md
│   └── solution2_card.md
├── outputs/                 # Target directory for generated Prediction CSVs
└── spec/                    # Coursework specification
```

## How to Run

### Setup

Ensure you install the dependencies:

```bash
pip install -r requirements.txt
```

### Training

Each solution has its own modular training scripts. See their respective nested READMEs for in-depth details.

```bash
# Solution 1 (Feature Extraction + Stacking)
python src/solution1/training/preprocess.py --input data/training_data/AV/train.csv --output data/processed_train.pkl
python src/solution1/training/feature_extraction.py --input data/processed_train.pkl --output src/solution1/features/all_features_train.npy
python src/solution1/training/train_base_models.py --features src/solution1/features/all_features_train.npy --output_dir models/solution1/

# Solution 2 (Example: DeBERTa Base Model)
python -m src.solution2.deberta_model.train
```

### Inference (Generating Final Submission)

To generate the final submission-ready CSVs (containing a single `prediction` column), use the top-level inference scripts created for each solution:

```bash
# Generate submission for Solution 1 (Assuming features are extracted)
python -m src.solution1.predict --features src/solution1/features/all_features_test.npy --output outputs/solution1/predictions_test.csv

# Generate submission for Solution 2 (Assuming base model probs are computed)
python -m src.solution2.predict --split test
```

### Evaluation (Standalone)

We enforce a strict separation between training and evaluation logic. The unified evaluation script can be run on any outputs against a gold standard CSV:

```bash
python src/evaluate.py \
    --predictions outputs/solution2/predictions_dev.csv \
    --gold data/training_data/AV/dev.csv
```

### Interactive Demo

Open the corresponding notebook in `notebooks/` and run all cells for an interactive walkthrough:

- `demo_solution1.ipynb` — Loads saved models, generates predictions using stacking.
- `demo_solution2.ipynb` — Loads saved Transformer ensemble, generates predictions.

## Model Cards & Cloud Links

Please refer to the `model_cards/` directory to read detailed specifications for both solutions:

- [Solution 1 Model Card](model_cards/solution1_card.md)
- [Solution 2 Model Card](model_cards/solution2_card.md)

Weights are saved externally due to GitHub file size limitations.

| Solution | Component | Cloud Link |
|----------|-----------|------------|
| Solution 1 | Corpus Vectors | [Google Drive](https://drive.google.com/file/d/1JFsQpv3RCfa0HEcwK4zz4gAVrWxDWUrg/view?usp=sharing) |
| Solution 1 | Model A2 (LightGBM) | [Google Drive](https://drive.google.com/file/d/1I9Cy6XddLRY8TUcs0VMh895WmySVKj45/view?usp=sharing) |
| Solution 2 | XLNet Base | [Google Drive](https://drive.google.com/file/d/1HWnOv36f-VOe92tKl2saQ1xs4k7hwbty/view?usp=sharing) |
| Solution 2 | RoBERTa Base | [Google Drive](https://drive.google.com/file/d/15M0kqGWtfLrO5F2mB3O4MH3Yt44dZDTZ/view?usp=sharing) |
| Solution 2 | ELECTRA Base | [Google Drive](https://drive.google.com/file/d/1ce5tgkHNuZzLhpoata9FiBDsPAtxGPKe/view?usp=sharing) |
| Solution 2 | DeBERTa Base | [Google Drive](https://drive.google.com/file/d/1IfzDd87FARs4CE7rEviKcvI2UL9sKH4s/view?usp=sharing) |
| Solution 2 | Meta-Learner | [Google Drive](https://drive.google.com/file/d/1jf9hXJtY0LCHW-dOiEBgo-SdnrZK0Hri/view?usp=sharing) |

## Attribution

- **Hugging Face Transformers**: Code for tokenization and model initialization heavily relies on the `transformers` library (Wolf et al., 2020).
- **Scikit-Learn**: Used for metric computation and core preprocessing inside the Stacking ensemble.
- **Pre-trained Models**: DeBERTa-v3-large, RoBERTa-large, ELECTRA-large, and XLNet-large pre-trained checkpoint weights are loaded identically from the official Hugging Face hub.

## Use of Generative AI Tools

- **LLM Assistance (Claude/Gemini via IDE)**: Used as an advanced autocomplete and restructuring agent. Functions included generating PyTorch DataLoader boilerplates, writing custom markdown tables for the Model Cards, refactoring directory structures uniformly, and providing boilerplate metric calculation functions. No core machine learning logic or novel architecture design was wholesale generated; AI served solely as an assistive programming layer.
