# Small Model (Bi-Encoder / Siamese Network)

This directory contains the code for **Model C2**, designed specifically to be trained locally on a Mac (using Apple Silicon MPS if available). 

It uses a Siamese Bi-Encoder architecture (defaulting to `all-MiniLM-L6-v2`), which is highly complementary to the Big Model's cross-encoder when it comes time to ensemble.

## Architecture

Unlike the big model which concatenates texts (`[CLS] text_1 [SEP] text_2 [SEP]`), this model:
1. Embeds `text_1` using the transformer.
2. Embeds `text_2` using the **same** transformer.
3. Calculates the difference `|u - v|` between the two embeddings.
4. Uses a final linear layer to predict authorship.

## How to Run locally

Since this is designed for local prototyping, there are no Slurm scripts here.

**To Train (Standard):**
```bash
python -m src.solution2.small_model.train
```

**To Run Automated Hyperparameter Tuning:**
```bash
python -m src.solution2.small_model.train_multiple
```
This script will loop over 4 different combinations of Learning Rate and Batch Size automatically, and save the best model from each into separate folders.

**To Predict (generate ensemble inputs):**
```bash
python -m src.solution2.small_model.predict
```
*Outputs to: `outputs/solution2/small_model/small_probs_val.csv`*

*(Note: Ensure you have run `pip install -r requirements.txt` so PyTorch and Transformers are installed).*
