#!/bin/bash --login

### Partition and GPU configuration
#SBATCH -p gpuL                   # gpuL for L40s nodes
#SBATCH -G 1                      # Request 1 GPU (inference is less demanding than training)
#SBATCH -t 01:00:00               # Wallclock limit: 1 hour (inference is fast)
#SBATCH -n 1                      # 1 task
#SBATCH -c 4                      # 4 CPU cores
#SBATCH -J deberta_model_predict   # Job name
#SBATCH -o logs/predict-%j.out    # Standard output log
#SBATCH -e logs/predict-%j.err    # Standard error log

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
REPO_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../../..")
cp -r "$REPO_ROOT" "$TMPDIR/repo"

# 5. Move into the NVMe *repository root*
cd "$TMPDIR/repo"

# 6. Run DeBERTa inference on train and dev sets
echo "Starting DeBERTa predictions..."

echo "Predicting on dev set..."
python -m src.solution2.deberta_model.predict \
    --checkpoint models/solution2/deberta_model/best_model.pt \
    --input data/training_data/AV/dev.csv \
    --split dev

echo "Predicting on train set..."
python -m src.solution2.deberta_model.predict \
    --checkpoint models/solution2/deberta_model/best_model.pt \
    --input data/training_data/AV/train.csv \
    --split train

# 7. Copy results back to the original repository
echo "Predictions complete. Copying results back..."
mkdir -p "$REPO_ROOT/outputs/solution2/deberta_model"
mkdir -p "$TMPDIR/repo/outputs/solution2/deberta_model"
cp "$TMPDIR/repo/outputs/solution2/deberta_model"/* "$REPO_ROOT/outputs/solution2/deberta_model/" 2>/dev/null || true

echo "Done! Prediction CSVs saved to $REPO_ROOT/outputs/solution2/deberta_model/"
