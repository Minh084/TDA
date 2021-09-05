#!/bin/bash
#SBATCH -J abl_model_training
#SBATCH -o /home/jupyter/ThickDesc/OutputTD/3_models/logs/get_sparse."%j".out

# Request 72 hour run time
#SBATCH -t 72:0:0
#SBATCH --mem=16g
#SBATCH -p normal

Features=(vitals)
feature=${Features[${SLURM_ARRAY_TASK_ID}-1]}
model=lightgbm
echo $label

dirO=/home/jupyter/ThickDesc/OutputTD/3_models/ablation/$feature/${label}/
model_file=/home/jupyter/ThickDesc/OutputTD/3_models/${label}/${model}/${model}_validation_params.json
mkdir -p $dirO

data_dir=/home/jupyter/ThickDesc/OutputTD/3_models/ablation/$feature/


python /home/jupyter/ThickDesc/TriageTD/3_models/train_model.py --model_class $model  --data_dir $data_dir  --label $label --output_dir $dirO --val 0 --model_file ${model_file}