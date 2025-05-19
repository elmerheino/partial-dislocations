#!/bin/bash
#SBATCH --time=06:30:00
#SBATCH --mem-per-cpu=20G
#SBATCH --output=depinning-noise-incremental.out
#SBATCH --array=1-100
#SBATCH --cpus-per-task=5
#SBATCH --job-name=noise-incremental
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

DATE=$(date +"%Y-%m-%d")
echo $DATE
NAME=2025-05-19-noise-incremental

module load scicomp-python-env

srun python3 findCriticalForce.py -id ${SLURM_ARRAY_TASK_ID} -len 100 -f ${WRKDIR}/${NAME} -c 5