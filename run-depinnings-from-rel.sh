#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --job-name=16-7-depinning
#SBATCH --mem-per-cpu=600M
#SBATCH --output=16-7-depinning.out
#SBATCH --cpus-per-task=20
#SBATCH --array=0-299
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

srun python3 depinningFromRel.py --folder ${WRKDIR}/14-7-relaksaatio/perfect --perfect --points 30 --cores 20 --task-id ${SLURM_ARRAY_TASK_ID} --time 300000 --dt 10 --out-folder ${WRKDIR}/15-7-tuloksia/single-dislocation