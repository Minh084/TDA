#!/bin/bash -l
# NOTE the -l flag!
#
#SBATCH -J get_sparse
#SBATCH -o /home/ccorbin/logs/get_sparse."%j".out

# Request 24 hour run time
#SBATCH -t 24:0:0
#SBATCH --mem=32gb
#SBATCH -p normal

source /share/sw/open/anaconda/3/bin/activate
which python
# source activate /home/ccorbin/envs/custom

Features=(Diagnosis Meds lab_orders lab_results vitals Imaging Procedures demo)
feature=${Features[${SLURM_ARRAY_TASK_ID}-1]}

out_path=/share/pi/jonc101/triage/results/ablation_experiments

mkdir -p ${dirO}

python /home/ccorbin/BMI212/notebooks/save_sparse_mats.py --output_path $out_path --ablated_feature_type $feature