#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --output=depinning-14-2-2025.out
#SBATCH --array=100-101
#SBATCH --cpus-per-task=1

# You may put the commands below:
module load scicomp-python-env

# Job step
srun python3 main.py --seed ${SLURM_ARRAY_TASK_ID}  -f ${WRKDIR}/results -tmin 2.85 -tmax 3.2 -p 50 -dt 0.05 -t 10000 --partial
srun python3 main.py --seed ${SLURM_ARRAY_TASK_ID}  -f ${WRKDIR}/results/15-feb/single-dislocation -tmin 2.85 -tmax 3.25 -p 25 -dt 0.05 -t 10000 --single