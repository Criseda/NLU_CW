#!/bin/bash
set -e

# Change to repo root
cd "$(dirname "$0")/../../.."

echo "=== 1. Starting NLU Setup ==="
export PYTHONPATH="$PWD/nlu_bundle-feature-unified-local-scorer:$PYTHONPATH"

echo "=== 2. Running Inference: MiniLM-L6 ==="
python3 -m src.solution2.bi_encoder.predict --split dev

echo "=== 3. Running Inference: DeBERTa (Original) ==="
python3 -m src.solution2.big_model.predict --split dev

echo "=== 4. Running Inference: DeBERTa (7931) ==="
python3 -m src.solution2.big_model.predict --split 7931 --checkpoint models/solution2/big_model/model_7931.pt

echo "=== 5. Running Inference: DeBERTa (803) ==="
# Strong previous model checkpoint
python3 -m src.solution2.big_model.predict --split 803 --checkpoint models/solution2/big_model/model_803.pt

echo "=== 6. Running Inference: DeBERTa (8180) ==="
# Our new best-performing model checkpoint
python3 -m src.solution2.big_model.predict --split 8180 --checkpoint models/solution2/big_model/model_8180.pt

echo "=== 7. Fusing All 5 Models ==="
# Combines all model predictions via soft-voting
python3 -m src.solution2.ensemble.predict_ensemble --split dev

echo "=== 8. Scoring ==="
python3 -m local_scorer.main --task av --prediction outputs/solution2/ensemble/ENSEMBLE_AV_dev.csv
