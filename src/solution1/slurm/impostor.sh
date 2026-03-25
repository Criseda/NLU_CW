#!/bin/bash --login
#SBATCH --job-name=av_impostor
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=multicore
#SBATCH --output=logs/impostor_%j.out

module load apps/binapps/anaconda3/2023.09
conda activate av_env

python src/solution1/training/compute_imposter.py \
    --input     data/training_data/AV/train.csv \
    --output    src/solution1/features/impostor_train.npy \
    --n_jobs    $SLURM_NTASKS \
    --n_trials  100 \
    --n_impostors 50
