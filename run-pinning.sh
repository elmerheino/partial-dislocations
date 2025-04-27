#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=15G
#SBATCH --output=depinning.out
#SBATCH --array=0-100
#SBATCH --cpus-per-task=20

DATE=$(date +"%Y-%m-%d")
echo $DATE
NAME="non-physical-params"

CORES=20
NOISE=1.0

TAU_MIN=0
TAU_MAX=10
POINTS=100

DT=0.05
TIME=10000

# You may put the commands below:
module load scicomp-python-env

# Seed count is array-max/noise points

# Job step partial dislocation
srun python3 main.py -f ${WRKDIR}/results/${DATE}-${NAME}/partial-dislocation -p ${POINTS} -dt ${DT} -c ${CORES} -t ${TIME} --partial \
    pinning --seed ${SLURM_ARRAY_TASK_ID} -R ${NOISE} -tmin ${TAU_MIN} -tmax ${TAU_MAX}

# Job step perfect dislocation
srun python3 main.py -f ${WRKDIR}/results/${DATE}-${NAME}/single-dislocation -p ${POINTS} -dt ${DT} -c ${CORES} -t ${TIME} --single \
    pinning --seed ${SLURM_ARRAY_TASK_ID} -R ${NOISE} -tmin ${TAU_MIN} -tmax ${TAU_MAX}