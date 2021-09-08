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

Features=(none) # vitals cannot do more than 1, otherwise, it only take the last one: (none  vitals) --> only vitals got done
feature=${Features[${SLURM_ARRAY_TASK_ID}-1]} # only for slurm job queue

echo $feature

out_path=/home/jupyter/ThickDesc/OutputTD/3_models/ablation
# dirO=/home/jupyter/ThickDesc/OutputTD/3_models/ablation/$features -- will give vitals/vitals

mkdir -p $out_path

python /home/jupyter/ThickDesc/TriageTD/3_models/save_sparse_mats.py --output_path $out_path --ablated_feature_type $feature