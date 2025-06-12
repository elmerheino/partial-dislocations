#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=par-region3
#SBATCH --mem-per-cpu=2G
#SBATCH --output=2025-06-07-region-3.out
#SBATCH --array=1-20
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi


DATE=$(date +"%Y-%m-%d")
echo $DATE
NAME=2025-06-07-region-3

CORES=20

# You may put the commands below:
module load scicomp-python-env

NOISE_MIN=-3
NOISE_MAX=-1

ARRAY_LEN=20   # SEEDS*NOISES=ARRAY_LEN for a square grid
SEEDS=1         # Seed count is array-max/noise points

TIME=250000
DT=0.1

TAU_POINTS=20   # How many external forces are tried per noise level to find the critical force

# Job step partial dislocation
srun python3 main.py -p ${TAU_POINTS} -dt ${DT} -t ${TIME} --partial -f ${WRKDIR}/${NAME}/partial-dislocation -c ${CORES} \
    grid --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} --array-length ${ARRAY_LEN} --seeds ${SEEDS}
