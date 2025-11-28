#!/bin/bash

### job options ###

#SBATCH -J pretrain_procupineVAE_defaults             # name
#SBATCH -N 1 						# all cores are on one node
#SBATCH --cpus-per-task=8                        # number of cpus
#SBATCH -t 8:00:00 			        # time 1 hour per job days	
#SBATCH --mem 100G 				    # memory
#SBATCH --ntasks=1
#SBATCH -p gpu --gres gpu:1
#SBATCH --output=./procupine_pretrain.out
#SBATCH --error=./procupine_pretrain.err

module load anaconda

source /oscar/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh
only need this first time:
conda create -n procupine python=3.11.9
conda activate procupine
#Only need to activate this once
python -m pip install -e .



### your commands here

python ./procyon/training/train_procupine.py 