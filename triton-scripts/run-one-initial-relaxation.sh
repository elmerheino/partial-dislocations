#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=initial-relaxation-4
#SBATCH --mem-per-cpu=500M
#SBATCH --output=initial-relaxation.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

TIME=$1
SYSTEM=$2

srun python3 ../initialRelaxations.py --rmin $3 --rmax $4 --rpoints $5 --seeds $6 --n ${SYSTEM} --length ${SYSTEM} --folder $8 -c 20 --d0 $7 $9