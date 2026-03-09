#!/bin/bash --login
#SBATCH --job-name=av_infotheory
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=multicore
#SBATCH --output=logs/infotheory_%j.out

module load apps/binapps/anaconda3/2023.09
conda activate av_env

python src/solution1/training/compute_unmasking.py \
    --input  data/training_data/AV/train.csv \
    --output src/solution1/features/unmasking_train.npy \
    --n_jobs $SLURM_NTASKS
