#!/bin/bash

### job options ###

#SBATCH -J pretrain_procupineVAE_defaults             # name
#SBATCH -N 1 						# all cores are on one node
#SBATCH -n 4                        # number of cpus
#SBATCH -t 8:00:00 			        # time 1 hour per job days	
#SBATCH --mem 100G 				    # memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gres gpu:1
#SBATCH --mem=64G
#SBATCH --output=./procupine_pretrain.out
#SBATCH --error=./procupine_pretrain.err

source .procupine_venve/bin/activate
python -m pip install -e .

### your commands here
python ../procyon/training/train_procupine.py 