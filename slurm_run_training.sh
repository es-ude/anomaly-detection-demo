#!/bin/bash

#SBATCH -J anomaly-detection                      # name of the job
#SBATCH --nodes=1                                 # number of compute nodes
#SBATCH --time=2-00:00:00                         # max. run-time
#SBATCH --partition=GPU-small                     # gpu partition, small or big
#SBATCH --gres=gpu:1                              # amount of gpus
#SBATCH --mail-type=ALL                           # all events are reported via e-mail
#SBATCH --mail-user=vorname.nachname@uni-due.de  # user's e-mail adress

ENV_NAME="anomaly-detection"

# Change to the directory the job was submitted from
cd $SLURM_SUBMIT_DIR

module load nvhpc/23.9

module load cuda/12.3.2

module load miniconda/3

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "$ENV_NAME"; then
    echo "conda venv '$ENV_NAME' already exists"
else
    echo "create conda venv '$ENV_NAME'"
    conda create -n "$ENV_NAME" --yes python=3.13
fi

conda activate "$ENV_NAME"

pip install uv

uv sync

uv run --env-file=.env experiments/train_cookie_autoencoder.py
