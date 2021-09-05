#!/bin/bash -l
# NOTE the -l flag!
#
#SBATCH -J get_sparse
#SBATCH -o /home/jupyter/ThickDesc/OutputTD/3_models/logs/get_sparse."%j".out

# Request 24 hour run time
#SBATCH -t 24:0:0
#SBATCH --mem=32gb
#SBATCH -p normal

which python

Features=(vitals)
feature=${Features[${SLURM_ARRAY_TASK_ID}-1]}


dirO=/home/jupyter/ThickDesc/OutputTD/3_models/ablation
# dirO=/home/jupyter/ThickDesc/OutputTD/3_models/ablation/$features -- will give vitals/vitals

mkdir -p $dirO

python /home/jupyter/ThickDesc/TriageTD/3_models/save_sparse_mats.py --output_path $dirO --ablated_feature_type $feature