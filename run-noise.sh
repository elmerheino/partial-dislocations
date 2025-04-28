#!/bin/bash
#SBATCH --time=06:30:00
#SBATCH --mem-per-cpu=9G
#SBATCH --output=depinning-noise.out
#SBATCH --array=1-100
#SBATCH --cpus-per-task=20

DATE=$(date +"%Y-%m-%d")
echo $DATE

CORES=20

# You may put the commands below:
module load scicomp-python-env

NOISE_MIN=-2
NOISE_MAX=2

ARRAY_LEN=100   # SEEDS*NOISES=ARRAY_LEN for a square grid
SEEDS=1         # Seed count is array-max/noise points

# Job step perfect dislocation
srun python3 main.py -p 100 -dt 0.05 -t 10000 --single -f ${WRKDIR}/results/${DATE}-noise/single-dislocation -c ${CORES} \
    grid --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} --array-length ${ARRAY_LEN} --seeds ${SEEDS}

# Job step partial dislocation
srun python3 main.py -p 100 -dt 0.05 -t 10000 --partial -f ${WRKDIR}/results/${DATE}-noise/partial-dislocation -c ${CORES} \
    grid --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} --array-length ${ARRAY_LEN} --seeds ${SEEDS}
