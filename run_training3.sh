#!/bin/bash
#SBATCH --job-name=Model_training_job
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition=toreador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=3
##SBATCH --mem-per-cpu=9625MB ##9.4GB, modify based on needs
#SBATCH -D /home/aidbucha

# Load required modules
module load gcc/9.3.0 openmpi/3.1.6-cuda

# Set up distributed training environment
export MASTER_ADDR="localhost"
export MASTER_PORT=12355
export WORLD_SIZE=3  # Number of processes (should match --nproc_per_node)

# Activate Conda environment correctly
export CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate FtestEnv
export PATH="/home/aidbucha/ENTER/envs/FtestEnv/bin:$PATH"

# Debug: Check Python version
echo "Using Python from: $(which python)"
echo "Python version: $(python --version)"

# Run the training script explicitly with the correct Python
/home/aidbucha/ENTER/envs/FtestEnv/bin/python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 train_model4.py
