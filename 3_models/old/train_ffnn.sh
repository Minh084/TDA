#!/bin/bash
#SBATCH -J ff_nn
#SBATCH -o /home/ccorbin/logs/ffnn_%A_%a.out

# Request 72 hour run time
#SBATCH -t 72:0:0
#SBATCH --mem=50g
#SBATCH --gres gpu:1
#SBATCH -p gpu

source /share/sw/open/anaconda/3/bin/activate
source activate /share/pi/jonc101/envs/custom2

Labels=(acute_to_critical_label  critical_to_acute_label  first_label  has_admit_label  label_24hr_recent  label_max24)
#Models=(elastic_net lasso random_forest lightgbm ridge)
label=${Labels[${SLURM_ARRAY_TASK_ID}-1]}
echo $label
model=ffnn
dirO=/share/pi/jonc101/triage/results/${label}/${model}/
mkdir -p $dirO

data_dir=/home/ccorbin/BMI212/data/

python /home/ccorbin/BMI212/notebooks/models/train_ffnn.py --data_dir $data_dir --label $label --output_dir $dirO 