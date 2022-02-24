#!/bin/bash -l
# NOTE the -l flag!
#
#SBATCH -J get_sparse
#SBATCH -o /home/jupyter/ThickDesc/OutputTD/6_validation/logs/get_sparse."%j".out

# Request 24 hour run time
#SBATCH -t 24:0:0
#SBATCH --mem=32gb
#SBATCH -p normal

which python

# check the print out size 43980 vs. 41366 

cohort=6_7_cohort4_all
echo $cohort

out_path=/home/jupyter/ThickDesc/OutputTD/6_validation/models/

mkdir -p $out_path

python /home/jupyter/ThickDesc/TriageTD/6_validation/6.12_sparse_matrix.py --output_path $out_path --cohort $cohort