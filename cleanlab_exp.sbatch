#!/bin/bash

#SBATCH --nodes=1               ## always 1 (we have just 1 server)
#SBATCH --job-name=cl_tac      ## name you give to your job
#SBATCH --output=cl_tac%j.out  ## sysout and syserr merged together
#SBATCH --cpus-per-task=2       ## number of cores for the job - max 80
#SBATCH --mem=128G                ## total memory for the job - max 512G
#SBATCH --gres=gpu:1            ## number of GPUs for the job - max :8

source /storage/sedovaa20/knodle_venv/bin/activate           ## activate ptg venv
cd /storage/sedovaa20/knodle
export PYTHONPATH="$(pwd)"
python3 examples/trainer/cleanlab/cleanlab_training_tutorial.py \
--path_to_data data_from_minio/tac/processed