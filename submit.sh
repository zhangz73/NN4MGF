#!/bin/bash
#SBATCH --job-name=empire_gpu
#SBATCH --partition=cornell
#SBATCH --account=cornell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1  # Request 1 GPU (max 8)
#SBATCH --time=24:00:00     # Time limit hrs:min:sec
#SBATCH --output=job_%j.out
#unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

# Run your command

srun -u python3 test_skew_symmetric.py 30

