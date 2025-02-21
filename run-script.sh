#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=depinning-14-2-2025.out
#SBATCH --array=100-101
#SBATCH --cpus-per-task=5

# You may put the commands below:
module load scicomp-python-env

# Job step
srun python3 main.py --seed ${SLURM_ARRAY_TASK_ID}  -f ${WRKDIR}/results/20-feb/partial-dislocation -tmin 2.9 -tmax 3.10 -p 100 -dt 0.05 -c 5 -t 10000 --partial
srun python3 main.py --seed ${SLURM_ARRAY_TASK_ID}  -f ${WRKDIR}/results/20-feb/single-dislocation -tmin 2.9 -tmax 3.10 -p 100 -dt 0.05 -c 5 -t 10000 --single
