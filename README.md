# COMP34812 NLU Shared Task

> **Track:** C — Authorship Verification

## Project Structure

```
NLU_CW/
├── data/                    # All data (trial, training, test)
├── eda/                     # Exploratory Data Analysis on each track
├── notebooks/
│   ├── demo_solutionA.ipynb # Inference demo — Solution A (Category C)
│   └── demo_solutionC.ipynb # Inference demo — Solution C (Category C)
├── src/
│   ├── solution1/           # Training code for Solution A
│   ├── solution2/           # Training code for Solution C
│   ├── evaluation/          # Evaluation scripts (F1, accuracy, etc.)
│   └── utils/               # Shared utilities (data loading, preprocessing)
├── models/                  # Saved model weights (large files on OneDrive)
├── model_cards/             # Model cards (markdown)
├── outputs/                 # Prediction CSVs
├── poster/                  # Poster PDF
└── spec/                    # Coursework specification
```

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
# Solution 1 (Category _)
python src/solution1/train.py

# Solution 2 (Category _)
python src/solution2/train.py
```

### Evaluation (Dev Set)
```bash
python src/evaluation/evaluate.py --predictions outputs/dev_preds_s1.csv --gold data/training_data/[TRACK]/dev.csv
```

### Inference (Demo)
Open the corresponding notebook in `notebooks/` and run all cells:
- `demo_solution1.ipynb` — loads saved model, generates predictions
- `demo_solution2.ipynb` — loads saved model, generates predictions

## Model Links

| Solution | Category | Cloud Link |
|----------|----------|------------|
| Solution 1 | _ | [OneDrive link] |
| Solution 2 | _ | [OneDrive link] |

## Attribution

- [List any reused code / data sources here]

## Use of Generative AI Tools

- [Declare any AI tools used and their purpose here]
