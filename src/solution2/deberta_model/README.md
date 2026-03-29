# DeBERTa Model (DeBERTa-v3-large)

This directory contains the code to train and run inference for the DeBERTa-v3-large cross-encoder for the NLU Authorship Verification (AV) track.

## File Overview

* **`config.py`**: The central configuration file. This contains all hyperparameters (batch size, learning rate, epochs) and all file paths. **Edit this file if you want to change training settings.**
* **`model.py`**: Contains the PyTorch `AVCrossEncoder` architecture class, which wraps the Hugging Face DeBERTa model with a classification head.
* **`train.py`**: The main script to train the model. It handles the dataset, tokenization, training loop, evaluation on the dev set, and gradient accumulation. 
* **`predict.py`**: The inference script. It takes a trained model checkpoint and a CSV file, runs predictions, and saves the probabilities.
* **`submit.slurm`**: A Slurm batch script to submit the training job to the CSF3 cluster (requesting 2x A100 GPUs).

---

## 🚀 How to Run

### 1. Training on the Cluster (CSF3)

To train the model using the cluster's GPUs, navigate to this directory in your terminal:
```bash
cd path/to/NLU_CW/src/solution2/deberta_model
```
Then submit the batch job:
```bash
sbatch submit.slurm
```

Then to see gpu usage:
```bash
ssh nodexyz #xyz being number of the node
module load tools/bintools/nvitop
nvitop
```

**What happens during training:**
1. The Slurm script copies the entire repository to the cluster's fast local NVMe storage (`$TMPDIR/repo`).
2. It trains the model using the parameters specified in `config.py`.
3. It saves the best model checkpoint based on the validation F1 score.
4. Once finished, it copies the saved models and logs back into the main repository at `models/solution2/deberta_model/`.

*(If you want to train locally or interactively, just run `python -m src.solution2.deberta_model.train` from the project root).*

### 2. Running Inference (Predictions)

To generate predictions using a trained model, run the `predict.py` script. Run this from the project root (`NLU_CW`):

**Predict on the default Dev Set (defined in config.py):**
```bash
python -m src.solution2.deberta_model.predict
```
*Outputs to: `outputs/solution2/deberta_model/probs_val.csv`*

**Predict on a custom Test Set:**
```bash
python -m src.solution2.deberta_model.predict \
    --input data/training_data/AV/test.csv \
    --checkpoint models/solution2/deberta_model/best_model.pt \
    --split test
```
*Outputs to: `outputs/solution2/deberta_model/probs_test.csv`*

---

## 📂 Where Does Everything Go?

The model respects a strict folder hierarchy to keep the repository clean.

### Inputs (Data)
The model expects the training and evaluation data to be located in:
* `data/training_data/AV/train.csv`
* `data/training_data/AV/dev.csv`
*(These paths are configured in `config.py`).*

### Model Checkpoints
During training, the highest-performing checkpoint (based on the `dev set` F1 score) is saved here:
* `models/solution2/deberta_model/best_model.pt`

### Inference Outputs
When you run `predict.py`, the predicted probabilities are saved here:
* `outputs/solution2/deberta_model/probs_val.csv` (or whatever `--split` name you provided).

These CSV probability files are designed to be explicitly loaded and used by the `ensemble` module later on!
