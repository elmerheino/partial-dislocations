#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --job-name=7-7-allkurelaksointi-testi
#SBATCH --mem-per-cpu=300M
#SBATCH --output=7-7-relaksointi.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

TIME=1000000
SYSTEM=512

srun python3 initialRelaxations.py --rmin -4 --rmax 0 --rpoints 20 --seeds 1 --time ${TIME} --n ${SYSTEM} --length ${SYSTEM} --dt 10 --folder ${WRKDIR}/7-7-relaksaatio/perfect -c 20