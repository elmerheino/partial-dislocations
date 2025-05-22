#!/bin/bash
#SBATCH --time=03:50:00
#SBATCH --job-name=noise-plot-perfect
#SBATCH --mem-per-cpu=20G
#SBATCH --output=2025-05-22-noise-cm-ivp-perfect.out
#SBATCH --array=1-1000
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi


DATE=$(date +"%Y-%m-%d")
echo $DATE
NAME=2025-05-22-noise-cm-ivp

CORES=20

# You may put the commands below:
module load scicomp-python-env

NOISE_MIN=-2
NOISE_MAX=1

ARRAY_LEN=1000   # SEEDS*NOISES=ARRAY_LEN for a square grid
SEEDS=10         # Seed count is array-max/noise points

TIME=10000
DT=0.025

# Job step perfect dislocation
srun python3 main.py -p 100 -dt ${DT} -t ${TIME} --single -f ${WRKDIR}/${NAME}/single-dislocation -c ${CORES} \
    grid --array-task-id ${SLURM_ARRAY_TASK_ID} --rmin ${NOISE_MIN} --rmax ${NOISE_MAX} --array-length ${ARRAY_LEN} --seeds ${SEEDS}