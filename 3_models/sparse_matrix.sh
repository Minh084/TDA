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

# check the print out size 43980 vs. 41366 

cohort=1_4_cohort_24hrpreadmit
# cohort=1_4_cohort
# cohort=1_5_cohort_final # or default
echo $cohort

out_path=/home/jupyter/ThickDesc/OutputTD/3_models/

mkdir -p $out_path

python /home/jupyter/ThickDesc/TriageTD/3_models/sparse_matrix.py --output_path $out_path --cohort $cohort