#!/bin/bash

#SBATCH --job-name="huggingface_gpu_job"
#SBATCH --output=job_%x-%j.out
#SBATCH --error=job_%x-%j.err
#SBATCH --partition="gpu"
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00 # 1 day, 0 hours, 0 minutes, 0 seconds
#SBATCH --reservation=mj6ux_122

python3 /p/llmreliability/test_repos/no_ensemble/llama_gsm8k.py