#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --job-name=alkurelaksointi-testi
#SBATCH --mem-per-cpu=2G
#SBATCH --output=relaksointi.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

srun python3 initialRelaxations.py --rmin -4 --rmax 0 --rpoints 20 --seeds 1 --time 1000000 --n 512 --length 512 --dt 10 --folder  ${WRKDIR}/4-7-relaksaatio/perfect -c 7