#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=depinning.out
#SBATCH --array=1-9
#SBATCH --cpus-per-task=9

DATE=$(date +"%Y-%m-%d")
echo $DATE

CORES=9

# You may put the commands below:
module load scicomp-python-env

NOISE_MIN=-2
NOISE_MAX=2

ARRAY_LEN=9 # SEEDS*SEEDS=ARRAY_LEN for a square grid
SEEDS=3

# Seed count is array-max/noise points

# Job step
srun python3 main.py --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} \
    --array-length ${ARRAY_LEN} --seeds ${SEEDS} -p 100 -dt 0.05 \
    -f ${WRKDIR}/results/${DATE}/single-dislocation -c ${CORES} \
    -t 10000 --single \

srun python3 main.py --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} \
    --array-length ${ARRAY_LEN} --seeds ${SEEDS} -p 100 -dt 0.05 \
    -f ${WRKDIR}/results/${DATE}/partial-dislocation -c ${CORES} \
    -t 10000 --partial \