#!/bin/bash --login

### Partition and GPU configuration
#SBATCH -p gpuL                   # gpuL selects L40s (48GB) partition
#SBATCH -G 1                      # Request 1 GPU
#SBATCH -t 12:00:00               # Wallclock limit: 9 hours
#SBATCH -n 1                      # 1 task
#SBATCH -c 4                      # 4 CPU cores (12 per GPU maximum)
#SBATCH -J roberta_model_nlu      # Job name
#SBATCH -o logs/roberta_train_%j.out      # Standard output log
#SBATCH -e logs/roberta_train_%j.err      # Standard error log

echo "Job running on node: $SLURMD_NODENAME"
echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_CPUS_PER_TASK CPU core(s)"

# 1. Clean environment and load modules
module purge
module load apps/binapps/conda/miniforge3/25.9.1
module load libs/cuda/12.4.1       # Recommended CUDA version for L40s

# 2. Activate Conda environment
conda activate nlu_big_model

# 3. Optimize multiprocessing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. Copy entire project root to fast NVMe storage ($TMPDIR)
echo "Copying project to NVMe storage ($TMPDIR)..."
# Assuming submission from src/solution2/roberta_model, the root is 3 levels up
REPO_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../../..")
cp -r "$REPO_ROOT" "$TMPDIR/repo"

# 5. Move into the NVMe *repository root* instead of the model folder
cd "$TMPDIR/repo"

# 6. Run the training script from the fast storage
echo "Starting RoBERTa training..."
# Run as a module from the root to fix relative imports: `from . import config`
python -m src.solution2.roberta_model.train_roberta

# 7. Copy the TRAINED MODEL back to the original repository (CRITICAL!)
echo "Training complete. Copying checkpoint back..."
mkdir -p "$REPO_ROOT/models/solution2/roberta_model"

# Copy ONLY the best_model.pt (not outputs, not logs)
if [ -f "$TMPDIR/repo/models/solution2/roberta_model/best_model.pt" ]; then
    cp "$TMPDIR/repo/models/solution2/roberta_model/best_model.pt" "$REPO_ROOT/models/solution2/roberta_model/"
    echo "✓ Checkpoint copied successfully to $REPO_ROOT/models/solution2/roberta_model/best_model.pt"
    ls -lh "$REPO_ROOT/models/solution2/roberta_model/best_model.pt"
else
    echo "ERROR: best_model.pt not found in $TMPDIR/repo/models/solution2/roberta_model/"
    echo "Directory contents:"
    ls -la "$TMPDIR/repo/models/solution2/roberta_model/" || echo "Directory doesn't exist"
    exit 1
fi

echo "--- Job finished at $(date) ---"
