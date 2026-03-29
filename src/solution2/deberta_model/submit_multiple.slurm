#!/bin/bash --login

### Partition and GPU configuration
#SBATCH -p gpuA                   # Partition (gpuA for A100, gpuL for L40S)
#SBATCH -G 1                      # Number of GPUs
#SBATCH -t 12:00:00               # Wallclock time (12 hours gives a safe buffer for 4 overnight models)
#SBATCH -n 1                      # 1 task
#SBATCH -c 4                      # Number of CPU cores (6 is optimal for 2 GPUs)
#SBATCH -J NLU_Sweep              # Job name
#SBATCH -o logs/sweep-%j.out      # Standard output log
#SBATCH -e logs/sweep-%j.err      # Standard error log

# 1. Clean environment and load modules
module purge
module load apps/binapps/conda/miniforge3/25.9.1
module load libs/cuda/12.4.1       # Recommended CUDA version for A100

# 2. Activate Conda environment
conda activate nlu_big_model

# 3. Optimize multiprocessing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 2. Logging and Project Location
echo "Starting job on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

# 4. Fast NVMe Storage Transfer ($TMPDIR)
echo "Copying project to NVMe storage ($TMPDIR)..."
REPO_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../../..")
cp -r "$REPO_ROOT" "$TMPDIR/repo"

cd "$TMPDIR/repo"

# 5. Execute the Grid Search Script
echo "Starting overnight grid search..."
python -m src.solution2.deberta_model.train_multiple
echo "Grid search complete."

# 6. Extract the precious models back to permanent storage!
echo "Copying sweep results back to central storage..."
mkdir -p "$REPO_ROOT/models/solution2/deberta_model"
mkdir -p "$REPO_ROOT/outputs/solution2/deberta_model"
mkdir -p "$TMPDIR/repo/models/solution2/deberta_model"
mkdir -p "$TMPDIR/repo/outputs/solution2/deberta_model"

cp -ru "$TMPDIR/repo/models/solution2/deberta_model/"* "$REPO_ROOT/models/solution2/deberta_model/"
cp -ru "$TMPDIR/repo/outputs/solution2/deberta_model/"* "$REPO_ROOT/outputs/solution2/deberta_model/"

echo "--- Job finished at $(date) ---"
