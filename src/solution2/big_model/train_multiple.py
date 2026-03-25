import os
import gc
import time
import torch
from src.solution2.big_model import config
from src.solution2.big_model.train import train

def main():
    """
    Overnight Hyperparameter Grid Search for DeBERTa-v3-large.
    Each configuration runs completely sequentially.
    Outputs are saved to dedicated subdirectories inside models/
    """
    
    # 1. Wallclock Safety Timer Setup
    # The submit_multiple.slurm provides a 12-hour window. 
    # We set a hard stop at 11.5 hours to guarantee plenty of time to copy NVMe files back.
    start_time = time.time()
    MAX_RUNTIME_HOURS = 11.5
    MAX_RUNTIME_SECONDS = MAX_RUNTIME_HOURS * 3600
    
    # 2. Focused Learning Rate Sweep
    # Narrow search for the optimal learning rate using the clean, unified baseline settings.
    configs_to_test = [
        {"lr": 3e-6, "warmup": 0.10, "batch": 8, "grad_accum": 2, "weight_decay": 0.01, "seed": 42, "name": "baseline_lr_3e6"},
        {"lr": 5e-6, "warmup": 0.10, "batch": 8, "grad_accum": 2, "weight_decay": 0.01, "seed": 42, "name": "baseline_lr_5e6"},
        {"lr": 7e-6, "warmup": 0.10, "batch": 8, "grad_accum": 2, "weight_decay": 0.01, "seed": 42, "name": "baseline_lr_7e6"},
        {"lr": 1e-5, "warmup": 0.10, "batch": 8, "grad_accum": 2, "weight_decay": 0.01, "seed": 42, "name": "baseline_lr_1e5"},
    ]

    base_save_dir = config.MODEL_SAVE_DIR
    base_out_dir  = config.OUTPUT_DIR

    results_summary = []

    for i, run in enumerate(configs_to_test):
        # 3. Wallclock Check before starting a new model
        elapsed_time = time.time() - start_time
        if elapsed_time > MAX_RUNTIME_SECONDS:
            print(f"\n[sweep] ⏰ WARNING: Elapsed time ({elapsed_time/3600:.2f}h) exceeds safety limit ({MAX_RUNTIME_HOURS}h).")
            print("[sweep] 🛑 Gracefully stopping the sweep so Slurm can copy the finished models back to permanent storage!")
            break
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
        config.SEED           = run["seed"]
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
