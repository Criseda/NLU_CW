#!/bin/bash --login

### Partition and GPU configuration
#SBATCH -p gpuL                   # gpuA selects A100 (80GB) partition, or gpuL for L40s partition
#SBATCH -G 2                      # Request 2 GPUs
#SBATCH -t 12:00:00               # Wallclock limit: 12 hours
#SBATCH -n 1                      # 1 task
#SBATCH -c 6                      # 4 CPU cores (12 per GPU maximum)
#SBATCH -J deberta_model_nlu       # Job name
#SBATCH -o logs/1continue-%j.out      # Standard output log
#SBATCH -e logs/1continue-%j.err      # Standard error log

echo "Job running on node: $SLURMD_NODENAME"
echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_CPUS_PER_TASK CPU core(s)"

# 1. Clean environment and load modules
module purge
module load apps/binapps/conda/miniforge3/25.9.1
module load libs/cuda/12.4.1       # Recommended CUDA version for A100

# 2. Activate Conda environment
conda activate nlu_big_model

# 3. Optimize multiprocessing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. Copy entire project root to fast NVMe storage ($TMPDIR)
echo "Copying project to NVMe storage ($TMPDIR)..."
# Assuming submission from src/solution2/deberta_model, the root is 3 levels up
REPO_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../../..")
cp -r "$REPO_ROOT" "$TMPDIR/repo"

# 5. Move into the NVMe *repository root* instead of the model folder
cd "$TMPDIR/repo"

# 6. Run the training script from the fast storage
echo "Starting training..."
# Run as a module from the root to fix relative imports: `from . import config`
python -m src.solution2.deberta_model.continue_baseline

# 7. Copy results back to the original repository 
echo "Training complete. Copying results back..."
# Ensure destination directories exist
mkdir -p "$REPO_ROOT/models/solution2/deberta_model"
mkdir -p "$REPO_ROOT/outputs/solution2/deberta_model"

# Create the temp output directories too, just in case the python script crashed before creating them
mkdir -p "$TMPDIR/repo/models/solution2/deberta_model"
mkdir -p "$TMPDIR/repo/outputs/solution2/deberta_model"

# Copy the newly created deberta_model folders back
cp -r "$TMPDIR/repo/models/solution2/deberta_model" "$REPO_ROOT/models/solution2/"
cp -r "$TMPDIR/repo/outputs/solution2/deberta_model" "$REPO_ROOT/outputs/solution2/"

echo "--- Job finished at $(date) ---"