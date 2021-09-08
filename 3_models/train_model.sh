#!/bin/bash
#SBATCH -J abl_model_training
#SBATCH -o /home/jupyter/ThickDesc/OutputTD/3_models/logs/get_sparse."%j".out

# Request 72 hour run time
#SBATCH -t 72:0:0
#SBATCH --mem=16g
#SBATCH -p normal

# Features=(none2  labs) #(none  vitals)
# feature=${Features[${SLURM_ARRAY_TASK_ID}-1]}

feature=none # none_ablated, meaning all features included
model=lightgbm
label=first_label
# label=death_24hr_recent_label
# val_flag=1 # training to get hyperparameters first

echo $feature
echo $model
echo $label
# echo $val_flag

dirO=/home/jupyter/ThickDesc/OutputTD/3_models/ablation_trainbinonly/$feature/${label}/
model_file=/home/jupyter/ThickDesc/OutputTD/3_models/ablation_trainbinonly/$feature/${label}/${model}_validation_params.json

mkdir -p $dirO

data_dir=/home/jupyter/ThickDesc/OutputTD/3_models/ablation/$feature/


# use val 1 for TRUE, run the training to get hyperparameters first
# use val 0 for FALSE, run the prediction model with chosen hyperparameters
python /home/jupyter/ThickDesc/TriageTD/3_models/train_model.py --model_class $model  --data_dir $data_dir  --label $label --output_dir $dirO --val 0 --model_file ${model_file}