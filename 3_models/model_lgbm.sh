#!/bin/bash
#SBATCH -J abl_model_training
#SBATCH -o /home/jupyter/ThickDesc/OutputTD/3_models/logs/get_sparse."%j".out

# Request 72 hour run time
#SBATCH -t 72:0:0
#SBATCH --mem=16g
#SBATCH -p normal

model=lightgbm
# cohort=1_4_cohort
cohort=1_5_cohort_final
# label=first_label # isn't this the default?
label=death_24hr_recent_label
# label=death_24hr_max_label

# val_flag=1
val_flag=0

echo $model
echo $cohort
echo $label
echo $val_flag

in_path=/home/jupyter/ThickDesc/OutputTD/3_models/$cohort/
out_path=/home/jupyter/ThickDesc/OutputTD/3_models/$cohort/${label}/
model_path=/home/jupyter/ThickDesc/OutputTD/3_models/$cohort/${label}/${model}_validation_params.json

mkdir -p $out_path

python /home/jupyter/ThickDesc/TriageTD/3_models/model_lgbm.py --model_class $model --data_dir $in_path  --label $label --output_dir $out_path --val $val_flag --model_file ${model_path}


# use val 1 for TRUE, run the training to get hyperparameters first
# use val 0 for FALSE, run the prediction model with chosen hyperparameters

# python /home/jupyter/ThickDesc/TriageTD/3_models/model_lgbm.py --model_class $model --data_dir $in_path  --label $label --output_dir $out_path --val 0 --model_file ${model_path}

# python /home/jupyter/ThickDesc/TriageTD/3_models/model_lgbm.py --model_class $model --data_dir $in_path  --label $label --output_dir $out_path --val 1 --model_file ${model_path}