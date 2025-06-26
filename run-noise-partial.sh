#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --job-name=par-sys256
#SBATCH --mem-per-cpu=1G
#SBATCH --output=2025-06-26-sys-256-partial.out
#SBATCH --array=1-10
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

# Kannattaa aloittaa ajaminen 30min ajalla, ja ne jotka timeouttaa niin sitten 1h 30 min menee läpi

DATE=$(date +"%Y-%m-%d")
echo $DATE
NAME=2025-06-26-sys-256

CORES=20

# You may put the commands below:
module load scicomp-python-env

NOISE_MIN=-3
NOISE_MAX=-1

ARRAY_LEN=10   # SEEDS*NOISES=ARRAY_LEN for a square grid
SEEDS=1         # Seed count is array-max/noise points

TIME=300000
DT=10
L=256

TAU_POINTS=20   # How many external forces are tried per noise level to find the critical force

# Job step partial dislocation
srun python3 main.py -p ${TAU_POINTS} -dt ${DT} -t ${TIME} --length ${L} --partial -f ${WRKDIR}/${NAME}/partial-dislocation -c ${CORES} \
    grid --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} --array-length ${ARRAY_LEN} --seeds ${SEEDS}
