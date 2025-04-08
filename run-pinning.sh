#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=depinning.out
#SBATCH --array=0-100
#SBATCH --cpus-per-task=9

DATE=$(date +"%Y-%m-%d")
echo $DATE
NAME="zaiser-params"

CORES=9
NOISE=1.0

TAU_MIN=0
TAU_MAX=10

POINTS=100
DT=0.005

# You may put the commands below:
module load scicomp-python-env

# Seed count is array-max/noise points

# Job step
srun python3 main.py -f ${WRKDIR}/results/${DATE}-${NAME}/partial-dislocation -p ${POINTS} -dt ${DT} -c ${CORES} -t 10000 --partial \
    pinning --seed ${SLURM_ARRAY_TASK_ID} -R ${NOISE} -tmin ${TAU_MIN} -tmax ${TAU_MAX}

srun python3 main.py -f ${WRKDIR}/results/${DATE}-${NAME}/single-dislocation -p ${POINTS} -dt ${DT} -c ${CORES} -t 10000 --single \
    pinning --seed ${SLURM_ARRAY_TASK_ID} -R ${NOISE} -tmin ${TAU_MIN} -tmax ${TAU_MAX}