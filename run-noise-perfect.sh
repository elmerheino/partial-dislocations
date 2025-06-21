#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=perfect-reg1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=2025-06-21-region-1-perfect.out
#SBATCH --array=1-10
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi


DATE=$(date +"%Y-%m-%d")
echo $DATE

NAME=2025-06-21-region-1

CORES=20

# You may put the commands below:
module load scicomp-python-env

NOISE_MIN=-3
NOISE_MAX=3

ARRAY_LEN=100   # SEEDS*NOISES=ARRAY_LEN for a square grid
SEEDS=1         # Seed count is array-max/noise points, so how many seeds per noise level

TIME=500000
DT=10

TAU_POINTS=100   # How many external forces are tried per noise level to find the critical force

# Job step perfect dislocation
srun python3 main.py -p ${TAU_POINTS} -dt ${DT} -t ${TIME} --single -f ${WRKDIR}/${NAME}/single-dislocation -c ${CORES} \
    grid --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} --array-length ${ARRAY_LEN} --seeds ${SEEDS}