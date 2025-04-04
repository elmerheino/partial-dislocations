#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=depinning.out
#SBATCH --array=0-100
#SBATCH --cpus-per-task=9

DATE=$(date +"%Y-%m-%d")
echo $DATE

CORES=9
NOISE=1.0

# You may put the commands below:
module load scicomp-python-env

# Seed count is array-max/noise points

# Job step
srun python3 main.py --seed ${SLURM_ARRAY_TASK_ID} -R ${NOISE} -f ${WRKDIR}/results/${DATE}-perfect-partial/partial-dislocation -tmin 2.65 -tmax 3.0 -p 100 -dt 0.05 -c ${CORES} -t 10000 --partial
srun python3 main.py --seed ${SLURM_ARRAY_TASK_ID} -R ${NOISE} -f ${WRKDIR}/results/${DATE}-perfect-partial/single-dislocation -tmin 2.65 -tmax 3.0 -p 100 -dt 0.05 -c ${CORES} -t 10000 --single