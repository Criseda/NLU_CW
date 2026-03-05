import os
import gc
import torch
from src.solution2.big_model import config
from src.solution2.big_model.train import train

def main():
    """
    Overnight Hyperparameter Grid Search for DeBERTa-v3-large.
    Each configuration runs completely sequentially.
    Outputs are saved to dedicated subdirectories inside models/
    """
    
    # Define our 4 targeted configs based on the successful 0.803 F1 run
    # Focus: Balancing High Recall (0.83) with Precision (0.77). All runs take ~110 mins each.
    configs_to_test = [
        # 1. The Baseline Reproduction (Confirm the 0.803 score logic)
        {"lr": 1e-5, "warmup": 0.10, "batch": 16, "grad_accum": 1, "weight_decay": 0.01, "name": "baseline_lr1e5_wd01"},
        
        # 2. High Regularization (Increase Weight Decay from 0.01 to 0.05 to penalize overconfident false positives)
        {"lr": 1e-5, "warmup": 0.10, "batch": 16, "grad_accum": 1, "weight_decay": 0.05, "name": "reg_lr1e5_wd05"},
        
        # 3. Smooth Explorer (Larger effective batch size to calculate a more precise gradient direction)
        {"lr": 1e-5, "warmup": 0.10, "batch": 16, "grad_accum": 2, "weight_decay": 0.01, "name": "smooth_lr1e5_bs32"},
        
        # 4. Meticulous Learner (Slightly lower LR, longer warmup for more cautious feature extraction)
        {"lr": 8e-6, "warmup": 0.15, "batch": 16, "grad_accum": 1, "weight_decay": 0.01, "name": "meticulous_lr8e6_wm15"},
    ]

    base_save_dir = config.MODEL_SAVE_DIR
    base_out_dir  = config.OUTPUT_DIR

    results_summary = []

    for i, run in enumerate(configs_to_test):
        print(f"\n{'='*70}")
        print(f"| Starting Grid Search Run {i+1}/{len(configs_to_test)}")
        print(f"| Name: {run['name']}")
        print(f"| Target LR: {run['lr']} | Target Warmup: {run['warmup']}")
        print(f"| Effective Batch: {run['batch'] * run['grad_accum']} (Batch {run['batch']} x Accum {run['grad_accum']})")
        print(f"{'='*70}\n")

        # 1. Dynamically override the global config module
        config.LEARNING_RATE  = run["lr"]
        config.WARMUP_RATIO   = run["warmup"]
        config.BATCH_SIZE     = run["batch"]
        config.GRAD_ACCUM     = run["grad_accum"]
        config.WEIGHT_DECAY   = run["weight_decay"]
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
        
        # 3. Clean up VRAM so the next model starts fresh seamlessly
        print("[sweep] Cleaning up GPU memory before next run...")
        gc.collect()
        torch.cuda.empty_cache()
    
    # 4. Print final sweep summary
    print(f"\n{'='*70}")
    print("GRID SEARCH COMPLETE. Final Summary:")
    for res in results_summary:
        print(f" - {res}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
