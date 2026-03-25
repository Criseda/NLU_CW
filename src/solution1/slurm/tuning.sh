#!/bin/bash --login
#SBATCH --job-name=av_tune_lgbm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpuA
#SBATCH --output=logs/tuning_lgbm_%j.out

module load apps/binapps/anaconda3/2023.09
conda activate av_env

python src/solution1/training/tune_hyperparams.py \
    --features src/solution1/features/all_features_train.npy \
    --labels   data/train_labels.npy \
    --model    lgbm \
    --n_trials 500 \
    --output   tuning/lgbm_study.db
