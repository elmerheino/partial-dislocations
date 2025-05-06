#!/bin/bash
#SBATCH --time=06:30:00
#SBATCH --mem-per-cpu=9G
#SBATCH --output=depinning-noise.out
#SBATCH --array=1-1000
#SBATCH --cpus-per-task=20

DATE=$(date +"%Y-%m-%d")
echo $DATE

CORES=20

# You may put the commands below:
module load scicomp-python-env

NOISE_MIN=-2
NOISE_MAX=2

ARRAY_LEN=100   # SEEDS*NOISES=ARRAY_LEN for a square grid
SEEDS=10         # Seed count is array-max/noise points

TIME=10000
DT=0.025

# Job step perfect dislocation
srun python3 main.py -p 100 -dt ${DT} -t ${TIME} --single -f ${WRKDIR}/results/${DATE}-noise/single-dislocation -c ${CORES} \
    grid --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} --array-length ${ARRAY_LEN} --seeds ${SEEDS}

# Job step partial dislocation
srun python3 main.py -p 100 -dt ${DT} -t ${TIME} --partial -f ${WRKDIR}/results/${DATE}-noise/partial-dislocation -c ${CORES} \
    grid --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} --array-length ${ARRAY_LEN} --seeds ${SEEDS}
