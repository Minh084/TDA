#!/bin/bash -l
# NOTE the -l flag!
#
#SBATCH -J get_sparse
#SBATCH -o /home/jupyter/ThickDesc/OutputTD/shc2021/logs/get_sparse."%j".out

# Request 24 hour run time
#SBATCH -t 24:0:0
#SBATCH --mem=32gb
#SBATCH -p normal

which python

cohort=14_cohort_final
echo $cohort

out_path=/home/jupyter/ThickDesc/OutputTD/shc2021/models2

mkdir -p $out_path

python /home/jupyter/ThickDesc/TriageTD/shc2021/15.2_sparse_matrix.py --output_path $out_path --cohort $cohort