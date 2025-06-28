#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --job-name=per-sys256
#SBATCH --mem-per-cpu=700M
#SBATCH --output=2025-06-26-sys-128-perfect.out
#SBATCH --array=1-10
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi


DATE=$(date +"%Y-%m-%d")
echo $DATE
NAME=2025-06-27-sys-128

CORES=20

# You may put the commands below:
module load scicomp-python-env

NOISE_MIN=-3
NOISE_MAX=-1

ARRAY_LEN=10   # SEEDS*NOISES=ARRAY_LEN for a square grid
SEEDS=1         # Seed count is array-max/noise points, so how many seeds per noise level

TIME=600000
DT=10
L=128

TAU_POINTS=20   # How many external forces are tried per noise level to find the critical force

# Job step perfect dislocation
srun python3 main.py -p ${TAU_POINTS} -dt ${DT} -t ${TIME} --length ${L} --single -f ${WRKDIR}/${NAME}/single-dislocation -c ${CORES} \
    grid --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} --array-length ${ARRAY_LEN} --seeds ${SEEDS}