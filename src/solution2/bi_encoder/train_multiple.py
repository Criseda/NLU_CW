import os
import gc
import torch
from src.solution2.small_model import config
from src.solution2.small_model.train import train

def main():
    """
    Hyperparameter Grid Search for the local Bi-Encoder Small Model.
    Each configuration runs sequentially on your Mac.
    Outputs are saved to dedicated subdirectories inside models/solution2/small_model/
    """
    
    # Define our targeted configs for the search
    configs_to_test = [
        # 1. The "Robust Regulariser" (Combatting the overfitting we saw)
        {"lr": 2e-5, "wd": 0.1, "warmup": 0.1, "batch": 16, "epochs": 5, "name": "robust_wd01"},
        
        # 2. The "Long Context" (Authorship signals often appear in longer text)
        {"lr": 1e-5, "wd": 0.05, "warmup": 0.15, "batch": 8, "epochs": 5, "maxlen": 512, "name": "long_context_512"},
        
        # 3. The "Patient Finetuner" (Very slow and steady)
        {"lr": 5e-6, "wd": 0.05, "warmup": 0.2, "batch": 16, "epochs": 8, "name": "patient_finetuner_8ep"},
        
        # 4. The "Large Batch Smooth" (Batch 32 + High Regularization)
        {"lr": 2e-5, "wd": 0.1, "warmup": 0.1, "batch": 32, "epochs": 5, "name": "smooth_bs32_wd01"},
    ]

    base_save_dir  = config.MODEL_SAVE_DIR
    base_out_dir   = config.OUTPUT_DIR
    default_epochs = config.EPOCHS
    default_maxlen = config.MAX_LENGTH

    results_summary = []

    for i, run in enumerate(configs_to_test):
        print(f"\n{'='*70}")
        print(f"| Starting Grid Search Run {i+1}/{len(configs_to_test)}")
        print(f"| Name: {run['name']}")
        print(f"| LR: {run['lr']} | WD: {run.get('wd', 0.01)} | MaxLen: {run.get('maxlen', default_maxlen)}")
        print(f"| Epochs: {run.get('epochs', default_epochs)} | Batch: {run['batch']}")
        print(f"{'='*70}\n")

        # 1. Dynamically override the global config module
        config.LEARNING_RATE  = run["lr"]
        config.WARMUP_RATIO   = run["warmup"]
        config.BATCH_SIZE     = run["batch"]
        config.EPOCHS         = run.get("epochs", default_epochs)
        config.WEIGHT_DECAY   = run.get("wd", 0.01)
        config.MAX_LENGTH     = run.get("maxlen", default_maxlen)
        config.MODEL_SAVE_DIR = os.path.join(base_save_dir, run["name"])
        config.OUTPUT_DIR     = os.path.join(base_out_dir, run["name"])

        # Create output directories for this specific run
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        # 2. Execute the self-contained training loop
        try:
            train()
            print(f"\n[sweep] ✓ Run {run['name']} completed successfully.")
            results_summary.append(f"{run['name']}: COMPLETED")
        except Exception as e:
            print(f"\n[sweep] ✗ Run {run['name']} FAILED with error: {e}")
            results_summary.append(f"{run['name']}: FAILED ({str(e)})")
        
        # 3. Clean up memory so the next model starts fresh
        print("[sweep] Cleaning up memory before next run...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    # 4. Print final sweep summary
    print(f"\n{'='*70}")
    print("GRID SEARCH COMPLETE. Final Summary:")
    for res in results_summary:
        print(f" - {res}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
