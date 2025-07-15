#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --job-name=depinning-rel
#SBATCH --mem-per-cpu=500M
#SBATCH --output=depinning-rel.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

srun python3 depinningFromRel.py --folder ${WRKDIR}/14-7-relaksaatio/perfect --cores 20 --perfect