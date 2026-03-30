#!/bin/bash --login

### Partition and GPU configuration
#SBATCH -p gpuA                   # GPU partition for neural network training
#SBATCH -G 1                      # 1 GPU (neural network meta-learner)
#SBATCH -t 00:30:00               # Wallclock limit: 30 minutes
#SBATCH -n 1                      # 1 task
#SBATCH -c 2                      # 2 CPU cores
#SBATCH -J ensemble_nlu           # Job name
#SBATCH -o logs/ensemble_%j.out   # Standard output log
#SBATCH -e logs/ensemble_%j.err   # Standard error log

echo "Job running on node: $SLURMD_NODENAME"
echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_CPUS_PER_TASK CPU core(s)"

# 1. Clean environment and load modules
module purge
module load apps/binapps/conda/miniforge3/25.9.1
module load libs/cuda/12.4.1       # CUDA for PyTorch

# 2. Activate Conda environment (create and install dependencies if not already done, this one is just an example)
conda activate nlu_big_model

# 3. Optimize multiprocessing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 4. Copy entire project root to fast NVMe storage ($TMPDIR)
echo "Copying project to NVMe storage ($TMPDIR)..."
REPO_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../../..")
cp -r "$REPO_ROOT" "$TMPDIR/repo"

# 5. Move into the NVMe *repository root*
cd "$TMPDIR/repo"

# 6. Calibrate base model predictions
echo "Calibrating base model predictions..."
python -m src.solution2.ensemble.calibrate_base_models

# 7. Train ensemble model (neural network meta-learner)
echo "Training neural network ensemble..."
python -m src.solution2.ensemble.train_ensemble

# 8. Make predictions on dev set with ensemble
echo "Generating ensemble predictions on dev set..."
python -m src.solution2.ensemble.predict_ensemble --split dev

echo "✓ Ensemble training and prediction complete"

# 8. Copy results back to original repository
echo "Copying ensemble model and predictions back to original repository..."
mkdir -p "$REPO_ROOT/models/solution2/ensemble"
mkdir -p "$REPO_ROOT/outputs/solution2/ensemble"
cp "$TMPDIR/repo/models/solution2/ensemble"/* "$REPO_ROOT/models/solution2/ensemble/" 2>/dev/null || true
cp "$TMPDIR/repo/outputs/solution2/ensemble"/* "$REPO_ROOT/outputs/solution2/ensemble/" 2>/dev/null || true
echo "✓ Ensemble model and predictions copied to $REPO_ROOT/"
