#!/bin/bash --login
#SBATCH --job-name=av_preprocess
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=multicore
#SBATCH --output=logs/preprocess_%j.out

module load apps/binapps/anaconda3/2023.09
conda activate av_env

python src/solution1/training/preprocess.py \
    --input  data/training_data/AV/train.csv \
    --output src/solution1/features/cached_spacy_train.pkl \
    --n_jobs $SLURM_NTASKS
