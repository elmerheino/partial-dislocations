#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=depinning.out
#SBATCH --array=1-100
#SBATCH --cpus-per-task=9

DATE=$(date +"%Y-%m-%d")
echo $DATE

CORES=9

# You may put the commands below:
module load scicomp-python-env

NOISE_MIN=-2
NOISE_MAX=2

# Seed count is array-max/noise points

# Job step
srun python3 main.py --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} \
    --array-length 100 --seeds 10 -p 100 -dt 0.05 \
    -f ${WRKDIR}/results/${DATE}/single-dislocation -c ${CORES} \
    -t 10000 --single \

srun python3 main.py --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} \
    --array-length 100 --seeds 10 -p 100 -dt 0.05 \
    -f ${WRKDIR}/results/${DATE}/partial-dislocation -c ${CORES} \
    -t 10000 --partial \