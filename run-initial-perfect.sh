#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=18-7-all-per-512
#SBATCH --mem-per-cpu=500M
#SBATCH --output=18-7-18-7-all-deltaR-per.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

TIME=100000
SYSTEM=512

srun python3 initialRelaxations.py --rmin -4 --rmax 4 --rpoints 100 --seeds 10 --time ${TIME} --n ${SYSTEM} --length ${SYSTEM} --dt 10 --folder ${WRKDIR}/18-7-all-deltaR/sys-512/perfect -c 20 --perfect