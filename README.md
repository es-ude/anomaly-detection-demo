# Anomaly Detection Demo

## Datasets

The MVTec AD dataset can be downloaded [[here]](https://www.mvtec.com/company/research/datasets/mvtec-ad). The cookie dataset is only available internally. Please contact @julianhoever or @florianhettstedt

## Environment

Example environment file (`.env`) to set all required environment variables:

```dotenv
DEVICE=<cpu/mps/cuda>
NUM_WORKERS=0

IMAGE_WIDTH=128
IMAGE_HEIGHT=128

MVTEC_DATASET_DIR=<absolute_path_to_dataset>
MVTEC_OBJECT=hazelnut
MVTEC_OUTPUT_DIR=<absolute_path_to_output_dir>/mvtec
MVTEC_CKPT_DIR=<absolute_path_to_project_dir>/src/demo/anomaly_detection/model_checkpoints/mvtec

COOKIE_AE_DATASET_DIR=<absolute_path_to_dataset>/v2
COOKIE_CLF_DATASET_DIR=<absolute_path_to_dataset>
COOKIE_OUTPUT_DIR=<absolute_path_to_output_dir>/cookie
COOKIE_CKPT_DIR=<absolute_path_to_project_dir>/src/demo/anomaly_detection/model_checkpoints/cookie

# ENABLE_PI_CAM=
# USE_CLASSIFIER=
```

## Run Experiments

### MVTec (Hazelnut) Training

```bash
uv run --env-file=.env src/demo/anomaly_detection/experiments/train_mvtec_autoencoder.py
```

### Cookie Training

```bash
uv run --env-file=.env src/demo/anomaly_detection/experiments/train_cookie_autoencoder.py
```

## Run Demo Application

> [!IMPORTANT]
>
> To use the Raspberry Camera Module system packages must be accessible.
> The `pycamera2` library and the `libcamera` implementation are required!
> This can be accomplished by creating the virtual environment with:
>
> ```bash
> uv venv --system-site-packages
> ```

You can start the demo with:

```bash
uv run src/demo_interface/demo.py
```

> [!NOTE]
>
> If you want to use a standard webcam instead of the raspberry camera module you need to update the `USE_RASPBERRY_CAMERA_MODULE` parameter in the [demo.py file](src/demo_interface/demo.py)

### Run on the Jetson Orin Nano with CUDA enabled

1. Enable `MAXN SUPER` user
2. Update System
3. Upgrade CUDA packages

- `wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb`
- `sudo dpkg -i cuda-keyring_1.1-1_all.deb`
- `sudo apt update`
- `sudo apt upgrade`

1. Install cuDSS: `sudo apt install cudss`
2. Clone and go to Project Repository
3. Start web-app: `uv run --python 3.10 --env-file=.env src/demo/web_app/main.py`

> [!CAUTION]
> Current Bug with Threading:
>
> ```
> Visit your app on one of these URLs: ObservableSet({'http://192.168.205.240:8080', 'http://localhost:8080', 'http://172.17.0.1:8080'})
> Process SpawnProcess-1:1:
> Traceback (most recent call last):
>   File "/usr/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
> ```

    self.run()

File "/usr/lib/python3.10/multiprocessing/process.py", line 108, in run
self.\_target(*self.\_args, \*\*self.\_kwargs)
File "/usr/lib/python3.10/concurrent/futures/process.py", line 240, in \_process_worker
call_item = call_queue.get(block=True)
File "/usr/lib/python3.10/multiprocessing/queues.py", line 122, in get
return \_ForkingPickler.loads(res)
File "/home/sid/Repositories/anomaly-detection-demo/.venv/lib/python3.10/site-packages/torch/multiprocessing/reductions.py", line 180, in rebuild_cuda_tensor
storage = storage_cls.\_new_shared_cuda(
File "/home/sid/Repositories/anomaly-detection-demo/.venv/lib/python3.10/site-packages/torch/storage.py", line 1464, in \_new_shared_cuda
return torch.UntypedStorage.\_new_shared_cuda(*args, \*\*kwargs)
torch.AcceleratorError: CUDA error: invalid argument
Search for `cudaErrorInvalidValue' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
Compile with`TORCH_USE_CUDA_DSA` to enable device-side assertions.

Task exception was never retrieved
future: <Task finished name='Task-8' coro=<DemoApplicationController.run() done, defined at /home/sid/Repositories/anomaly-detection-demo/src/demo/web_app/controller/demo_application_controller.py:32> exception=BrokenProcessPool('A process in the process pool was terminated abruptly while the future was running or pending.')>
Traceback (most recent call last):
File "/home/sid/Repositories/anomaly-detection-demo/src/demo/web_app/controller/demo_application_controller.py", line 35, in run
processed_frame = await self.\_take_and_process_frame()
File "/home/sid/Repositories/anomaly-detection-demo/src/demo/web_app/controller/demo_application_controller.py", line 51, in \_take_and_process_frame
return await run.cpu_bound(self.\_image_processor.process, frame)
File "/home/sid/Repositories/anomaly-detection-demo/.venv/lib/python3.10/site-packages/nicegui/run.py", line 99, in cpu_bound
raise e
File "/home/sid/Repositories/anomaly-detection-demo/.venv/lib/python3.10/site-packages/nicegui/run.py", line 88, in cpu_bound
return await \_run(process_pool, safe_callback, callback, *args, \*\*kwargs)
File "/home/sid/Repositories/anomaly-detection-demo/.venv/lib/python3.10/site-packages/nicegui/run.py", line 65, in \_run
return await loop.run_in_executor(executor, partial(callback,*args, \*\*kwargs))
concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.

> ```
>
> ```

## Training on AmplitUDE HPC

### Zip Folder

```bash
zip -r anomaly-detection-demo.zip anomaly-detection-demo -x anomaly-detection-demo/.venv/\*
```

This command zips the entire project folder, while excluding the venv directory and its contents.

### Data Storage Concept

/lustre/hpc_home/<unikennun>/\: permanent project data (quota=0.5TB)

/lustre/scratch/<custom-workspace>/: temporary working data (quota=10TB), needs to be created via [workspaces](https://escience-wissr.gitpages.uni-due.de/hpc-support/content/general/workspace.html)

/homes/<unikennung>/: university wide home storage

### Copy Files to HPC

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

### Jobscript (with slurm)

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

uv run --env-file=.env python -u src/demo/anomaly_detection/experiments/cookie/train_autoencoder.py
```

### Run Job (with slurm)

submit job: sbatch jobscript.sh </br>
queue overview: squeue -l </br>
cancel job: scancel <job_id> </br>
