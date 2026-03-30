#!/bin/bash --login

### Partition and GPU configuration
#SBATCH -p gpuA                   # gpuL selects L40s (48GB) partition
#SBATCH -G 1                      # Request 1 GPU (inference is less demanding than training)
#SBATCH -t 01:00:00               # Wallclock limit: 2 hours
#SBATCH -n 1                      # 1 task
#SBATCH -c 4                      # 4 CPU cores
#SBATCH -J xlnet_predict_nlu      # Job name
#SBATCH -o logs/predict_%j.out    # Standard output log
#SBATCH -e logs/predict_%j.err    # Standard error log

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
# Assuming submission from src/solution2/xlnet_model, the root is 3 levels up
REPO_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../../..")
cp -r "$REPO_ROOT" "$TMPDIR/repo"

# 5. Move into the NVMe *repository root*
cd "$TMPDIR/repo"

# 6. Run prediction scripts from the fast storage
echo "Starting XLNet inference..."

# Run prediction on dev set
echo "Running inference on dev set..."
python -m src.solution2.xlnet_model.predict \
    --input data/training_data/AV/dev.csv \
    --split dev

# Run prediction on train set (if needed for additional validation)
if [ -f data/training_data/AV/train.csv ]; then
    echo "Running inference on train set..."
    python -m src.solution2.xlnet_model.predict \
        --input data/training_data/AV/train.csv \
        --split train
fi

echo "✓ Inference complete"

# 7. Copy predictions back to original repository
echo "Copying predictions back to original repository..."
mkdir -p "$REPO_ROOT/outputs/solution2/xlnet_model"
cp "$TMPDIR/repo/outputs/solution2/xlnet_model"/* "$REPO_ROOT/outputs/solution2/xlnet_model/" 2>/dev/null || true
echo "✓ Predictions copied to $REPO_ROOT/outputs/solution2/xlnet_model/"
