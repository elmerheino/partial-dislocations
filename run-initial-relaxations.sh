#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=6-7-allkurelaksointi-testi
#SBATCH --mem-per-cpu=700M
#SBATCH --output=relaksointi.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

TIME=1000000

srun python3 initialRelaxations.py --rmin -4 --rmax 0 --rpoints 20 --seeds 1 --time ${TIME} --n 512 --length 512 --dt 10 --folder ${WRKDIR}/6-7-relaksaatio/perfect -c 20