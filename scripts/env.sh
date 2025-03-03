#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=install_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:59:58
#SBATCH --output=install_%A.out


module purge

module load 2024
module load Anaconda3/2024.06-1

cd ..
cd setup

conda env create -f env.yml
source activate aig-ml
pip install -r requirements.txt

echo "Environment created and activated"