#!/bin/bash
#SBATCH --job-name=foldflow_train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --partition=defq
#SBATCH -- export=ALL

cd /workspace/foldflow/
export PYTHONPATH=$(pwd)
python runner/train.py
