#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=23-7-strong-coupling
#SBATCH --mem-per-cpu=100M
#SBATCH --output=23-7-strong-coupling.out
#SBATCH --cpus-per-task=9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

TIME=$1
SYSTEM=$2

srun python3 ../initialRelaxations.py --rmin $3 --rmax $4 --rpoints $5 --seeds $6 --n ${SYSTEM} --length ${SYSTEM} --folder $8 -c 9 --d0 $7 $9 --only-fire