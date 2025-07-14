#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --job-name=7-7-allkurelaksointi-testi
#SBATCH --mem-per-cpu=500M
#SBATCH --output=14-7-relaksointi.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

TIME=100000
SYSTEM=512

srun python3 initialRelaxations.py --rmin -4 --rmax 0 --rpoints 30 --seeds 10 --time ${TIME} --n ${SYSTEM} --length ${SYSTEM} --dt 10 --folder ${WRKDIR}/14-7-relaksaatio/perfect -c 20