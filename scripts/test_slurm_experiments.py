#!/bin/bash
# Script to parallelize experiments with SLURM

# Name of the job
#SBATCH -J test-pyDML-Stats-experiments

# Queue
#SBATCH -p muylarga

# Output
#SBATCH -o test-pyDML-Stats-%A-%a.out

# Error
#SBATCH -e test-pyDML-Stats-%A-%a.err

# Tasks
#SBATCH --array=0-20

######################
# Begin work section #
######################

echo "pyDML-Stats test experiment id: $SLURM_ARRAY_TASK_ID"

MOD[0]="basic"
MOD[1]="ncm"
MOD[2]="ker"
MOD[3]="dim"

python ${MOD[$SLURM_ARRAY_TASK_ID]}.py test
python recopilate.py ${MOD[$SLURM_ARRAY_TASK_ID]} test