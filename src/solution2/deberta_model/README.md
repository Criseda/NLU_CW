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

### Training on the Cluster (CSF3)

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
