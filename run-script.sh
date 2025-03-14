#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=depinning-14-3-2025.out
#SBATCH --array=0-1
#SBATCH --cpus-per-task=5

# You may put the commands below:
module load scicomp-python-env

# Job step
srun python3 main.py --seed ${SLURM_ARRAY_TASK_ID}  -f ${WRKDIR}/results/14-mar/partial-dislocation -tmin 2.65 -tmax 3.0 -p 100 -dt 0.05 -c 5 -t 10000 --partial
srun python3 main.py --seed ${SLURM_ARRAY_TASK_ID}  -f ${WRKDIR}/results/14-mar/single-dislocation -tmin 2.65 -tmax 3.0 -p 100 -dt 0.05 -c 5 -t 10000 --single
