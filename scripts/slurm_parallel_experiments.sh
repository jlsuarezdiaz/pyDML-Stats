#!/bin/bash
# Script to parallelize experiments with SLURM

# Name of the job
#SBATCH -J pyDML-Stats-experiments

# Queue
#SBATCH -p muylarga

# Output
#SBATCH -o pyDML-Stats-%A-%a.out

# Error
#SBATCH -e pyDML-Stats-%A-%a.err

# Tasks
#SBATCH --array=0-20

######################
# Begin work section #
######################

echo "pyDML-Stats experiment id: $SLURM_ARRAY_TASK_ID"

ARGS[0]="small"
ARGS[1]="medium"
ARGS[2]="large1"
ARGS[3]="large2"
ARGS[4]="large3"
ARGS[5]="large4"
ARGS[6]="small"
ARGS[7]="medium"
ARGS[8]="large1"
ARGS[9]="large2"
ARGS[10]="large3"
ARGS[11]="large4"
ARGS[12]="small"
ARGS[13]="medium"
ARGS[14]="large1"
ARGS[15]="large2"
ARGS[16]="large3"
ARGS[17]="large4"
ARGS[18]="0"
ARGS[19]="1"
ARGS[20]="2"


MOD[0]="basic"
MOD[1]="basic"
MOD[2]="basic"
MOD[3]="basic"
MOD[4]="basic"
MOD[5]="basic"
MOD[6]="ncm"
MOD[7]="ncm"
MOD[8]="ncm"
MOD[9]="ncm"
MOD[10]="ncm"
MOD[11]="ncm"
MOD[12]="ker"
MOD[13]="ker"
MOD[14]="ker"
MOD[15]="ker"
MOD[16]="ker"
MOD[17]="ker"
MOD[18]="dim"
MOD[19]="dim"
MOD[20]="dim"

python3.6 ${MOD[$SLURM_ARRAY_TASK_ID]}.py ${ARGS[$SLURM_ARRAY_TASK_ID]}
python3.6 recopilate.py ${MOD[$SLURM_ARRAY_TASK_ID]} ${ARGS[$SLURM_ARRAY_TASK_ID]}

