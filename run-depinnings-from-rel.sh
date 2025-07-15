#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=depinning-2
#SBATCH --mem-per-cpu=500M
#SBATCH --output=depinning-rel.out
#SBATCH --cpus-per-task=20
#SBATCH --array=0-299
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

srun python3 depinningFromRel.py --folder ${WRKDIR}/14-7-relaksaatio-try-2/perfect --perfect --points 30 --cores 20 --task-id ${SLURM_ARRAY_TASK_ID} --time 100000 --dt 10