#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=3500M
#SBATCH --output=depinning-14-2-2025.out
#SBATCH --array=110-210
#SBATCH --cpus-per-task=20

# You may put the commands below:
module load scicomp-python-env

# Job step
srun python3 main.py --seed ${SLURM_ARRAY_TASK_ID}  -f ${WRKDIR}/results -tmin 2.85 -tmax 3.2 -p 50 -dt 0.05 -t 10000 -c 10
