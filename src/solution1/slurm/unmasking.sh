#!/bin/bash --login
#SBATCH --job-name=av_unmasking
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=multicore
#SBATCH --output=logs/unmasking_%j.out

module load apps/binapps/anaconda3/2023.09
conda activate av_env

python src/solution1/training/compute_unmasking.py \
    --input  data/training_data/AV/train.csv \
    --output src/solution1/features/unmasking_train.npy \
    --n_jobs $SLURM_NTASKS
