#!/bin/bash --login
#SBATCH --job-name=av_features
#SBATCH --ntasks=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=multicore
#SBATCH --output=logs/feature_extraction_%j.out

module load apps/binapps/anaconda3/2023.09
conda activate av_env

python src/solution1/training/feature_extraction.py \
    --input  src/solution1/features/cached_spacy_train.pkl \
    --output src/solution1/features/stylometric_train.npy \
    --n_jobs $SLURM_NTASKS
