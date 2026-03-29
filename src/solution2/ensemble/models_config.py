"""
models_config.py — Specify which models to include in the ensemble.

Simply add or remove models here, and they will automatically be included
in the ensemble training and prediction pipeline.

Format:
    'model_name': {
        'train_probs': 'path/to/train_probs.csv',
        'dev_probs': 'path/to/dev_probs.csv',
    }
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs", "solution2")

# ── ADD OR REMOVE MODELS HERE ───────────────────────────────────────────────
# Simply define the train and dev prediction file paths for each model
MODELS = {
    'roberta': {
        'train_probs': os.path.join(OUTPUTS_DIR, "roberta_model", "probs_train.csv"),
        'dev_probs': os.path.join(OUTPUTS_DIR, "roberta_model", "probs_dev.csv"),
    },
    'xlnet': {
        'train_probs': os.path.join(OUTPUTS_DIR, "xlnet_model", "probs_train.csv"),
        'dev_probs': os.path.join(OUTPUTS_DIR, "xlnet_model", "probs_dev.csv"),
    },
    'electra': {
        'train_probs': os.path.join(OUTPUTS_DIR, "electra_model", "probs_train.csv"),
        'dev_probs': os.path.join(OUTPUTS_DIR, "electra_model", "probs_dev.csv"),
    },
    'deberta': {
        'train_probs': os.path.join(OUTPUTS_DIR, "deberta_model", "probs_train.csv"),
        'dev_probs': os.path.join(OUTPUTS_DIR, "deberta_model", "probs_dev.csv"),
    },
    # 'minilm': {
    #     'train_probs': os.path.join(OUTPUTS_DIR, "minilm_model", "probs_train.csv"),
    #     'dev_probs': os.path.join(OUTPUTS_DIR, "minilm_model", "probs_dev.csv"),
    # },
    # To add a new model, simply add below:
    # 'model_name': {
    #     'train_probs': os.path.join(OUTPUTS_DIR, "model_name", "probs_train.csv"),
    #     'dev_probs': os.path.join(OUTPUTS_DIR, "model_name", "probs_dev.csv"),
    # },
}

# ── Training data (for labels) ──────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT_DIR, "data", "training_data", "AV")
TRAIN_LABELS_FILE = os.path.join(DATA_DIR, "train.csv")
DEV_LABELS_FILE = os.path.join(DATA_DIR, "dev.csv")
