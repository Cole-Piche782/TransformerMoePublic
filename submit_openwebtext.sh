#!/bin/bash
#SBATCH --job-name=train_router   # Job name
#SBATCH --output=slurm_output-1.txt        # SLURM output file
#SBATCH --ntasks=1                       # Number of tasks (usually 1)
#SBATCH --gpus-per-node=a100:4
#SBATCH --time=48:00:00                  # Time limit (2 days)
#SBATCH --mem=256GB                        # Memory limit

source ~/envs/moe/bin/activate

torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py -1 > program_output-1.txt