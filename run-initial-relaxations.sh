#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --job-name=18-7fire-plus-ivp-partial
#SBATCH --mem-per-cpu=500M
#SBATCH --output=18-7fire-plus-ivp.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

TIME=100000
SYSTEM=512

srun python3 initialRelaxations.py --rmin -4 --rmax 0 --rpoints 30 --seeds 1 --time ${TIME} --n ${SYSTEM} --length ${SYSTEM} --dt 10 --folder ${WRKDIR}/17-7-relaksaatio/partial -c 20 --partial