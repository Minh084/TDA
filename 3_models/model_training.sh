#!/bin/bash
#SBATCH -J abl_model_training
#SBATCH -o /home/ccorbin/logs/test/ablation_model_training."%j".out

# Request 72 hour run time
#SBATCH -t 72:0:0
#SBATCH --mem=16g
#SBATCH -p normal

source /share/sw/open/anaconda/3/bin/activate
source activate /share/pi/jonc101/envs/custom2

 
Features=(demo  Diagnosis  Imaging  lab_orders  lab_results  Meds  Procedures  vitals)
feature=${Features[${SLURM_ARRAY_TASK_ID}-1]}
model=lightgbm
echo $label

dirO=/share/pi/jonc101/triage/results/ablation_experiments/$feature/${label}/
model_file=/share/pi/jonc101/triage/results/${label}/${model}/${model}_validation_params.json
mkdir -p $dirO

data_dir=/share/pi/jonc101/triage/results/ablation_experiments/$feature/


python /home/ccorbin/BMI212/notebooks/models/train_model.py --model_class $model  --data_dir $data_dir  --label $label --output_dir $dirO --val 0 --model_file ${model_file}