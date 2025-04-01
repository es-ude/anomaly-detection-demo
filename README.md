# Anomaly Detection Demo

## Datasets

The MVTec AD dataset can be downloaded [[here]](https://www.mvtec.com/company/research/datasets/mvtec-ad). The cookie dataset is only available internally. Please contact @julianhoever or @florianhettstedt

## Environment

Example environment file (`.env`) to set all required environment variables:

```bash
DEVICE=<e.g. cuda, mps or cpu>
NUM_WORKERS=0

IMAGE_WIDTH=128
IMAGE_HEIGHT=128

MVTEC_DATASET_DIR=<path to mvtec dataset>
MVTEC_OBJECT=hazelnut

MVTEC_SAVED_MODEL=<path to mvtec outputs>/model.pt
MVTEC_HISTORY_FILE=<path to mvtec outputs>/history.csv
MVTEC_VERSION_FILE=outputs/mvtec<path to mvtec outputs>/commit_hash.txt

COOKIE_DATASET_DIR=<path to cookie dataset>

COOKIE_SAVED_MODEL=<path to cookie outputs>/model.pt
COOKIE_HISTORY_FILE=<path to cookie outputs>/history.csv
COOKIE_VERSION_FILE=<path to cookie outputs>/commit_hash.txt
```

## Run Experiments

### MVTec (Hazelnut) Training
```bash
uv run --env-file=.env experiments/train_mvtec_autoencoder.py
```

### Cookie Training
```bash
uv run --env-file=.env experiments/train_cookie_autoencoder.py
```

# Training on AmplitUDE HPC

## Zip Folder
```bash
zip -r anomaly-detection-demo.zip anomaly-detection-demo -x anomaly-detection-demo/.venv/\*
```
This command zips the entire project folder, while excluding the venv directory and its contents.

## Data Storage Concept

/lustre/hpc_home/<unikennun>/\: permanent project data (quota=0.5TB)

/lustre/scratch/<custom-workspace>/: temporary working data (quota=10TB), needs to be created via [workspaces](https://escience-wissr.gitpages.uni-due.de/hpc-support/content/general/workspace.html)

/homes/<unikennung>/: university wide home storage

## Copy Files to HPC

RSYNC
```bash
rsync -avz --exclude '.venv' /path/to/local/directory <unikennung>@gateway.amplitude.uni-due.de:/lustre/scratch/<your-workspace>/

The -a option ensures the file permissions and timestamps are preserved.
The -v option increases verbosity so you can monitor the transfer.
The -z option compresses data during transfer to speed up the process.
```

scp
```bash
scp /path/to/local/file <unikennung>@gateway.amplitude.uni-due.de:/lustre/scratch/<your-workspace>/
```

## Jobscript (with slurm)

```bash
#!/bin/bash

#SBATCH -J job-name                               # name of the job
#SBATCH --nodes=1                                 # number of compute nodes
#SBATCH --time=2-00:00:00                         # max. run-time
#SBATCH --partition=GPU-big                       # gpu partition, small or big
#SBATCH --gres=gpu:4                              # amount of gpus
#SBATCH --mail-type=ALL                           # all events are reported via e-mail
#SBATCH --mail-user=vorname.nachname@uni-due.de   # user's e-mail adress

ENV_NAME="venv-name"

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

uv run --env-file=.env python -u experiments/train_cookie_autoencoder.py
```

## Run Job (with slurm)

submit job: sbatch jobscript.sh </br>
queue overview: squeue -l </br>
cancel job: scancel <job_id> </br>





