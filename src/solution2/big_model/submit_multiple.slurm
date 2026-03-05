#!/bin/bash --login

### Partition and GPU configuration
#SBATCH -p gpuA                   # Partition (gpuA for A100, gpuL for L40S)
#SBATCH -G 1                      # Number of GPUs
#SBATCH -t 12:00:00               # Wallclock time (12 hours gives a safe buffer for 4 overnight models)
#SBATCH -n 1                      # 1 task
#SBATCH -c 4                      # Number of CPU cores (4 is optimal for 1 GPU)
#SBATCH -J NLU_Sweep              # Job name
#SBATCH -o logs/sweep-%j.out      # Standard output log
#SBATCH -e logs/sweep-%j.err      # Standard error log

# 1. Environment Setup
module load miniforge3/24.3.0-py3.10
source activate nlu_big_model

# 2. Logging and Project Location
echo "Starting job on node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

# 3. Fast NVMe Storage Transfer ($TMPDIR)
# Move code and data to local node storage to prevent IO bottlenecks
echo "Copying project to NVMe storage ($TMPDIR/repo)..."
mkdir -p $TMPDIR/repo
cp -r src data models outputs requirements.txt .env $TMPDIR/repo/
cd $TMPDIR/repo

# 4. Execute the Grid Search Script
echo "Starting overnight grid search..."
python -m src.solution2.big_model.train_multiple
echo "Grid search complete."

# 5. Extract the precious models back to permanent storage!
echo "Copying sweep results back to central storage..."
cd $SLURM_SUBMIT_DIR
cp -ru $TMPDIR/repo/models/solution2/big_model/* models/solution2/big_model/
cp -ru $TMPDIR/repo/outputs/solution2/big_model/* outputs/solution2/big_model/

echo "Submission script finished gracefully."
