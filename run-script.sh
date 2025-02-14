#!/bin/bash
#SBATCH --time=01:15:00
#SBATCH --mem-per-cpu=3500M
#SBATCH --output=array_example_%A_%a.out
#SBATCH --array=100-150
#SBATCH --cpus-per-task=10

# You may put the commands below:

# Job step
srun python main.py -s ${SLURM_ARRAY_TASK_ID} -tmin 2.85 -tmax 3.2 -p 50 -dt 0.05 -t 10000 -c 10