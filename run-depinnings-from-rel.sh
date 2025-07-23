#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --job-name=23-7-depinning
#SBATCH --mem-per-cpu=300M
#SBATCH --output=23-7-depinning.out
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

# srun python3 depinningFromRel.py --folder ${WRKDIR}/22-7-ihan-himona-dataa-vain-FIRE/l-256/perfect --perfect --points 20 --cores 20 --task-id ${SLURM_ARRAY_TASK_ID} --time 300000 --dt 100 --out-folder ${WRKDIR}/22-7-tuloksia/perfect/l-256

srun python3 depinningFromRel.py --folder ${WRKDIR}/22-7-ihan-himona-dataa-vain-FIRE/l-256/perfect --perfect --points 20 --cores 20 --task-id 0 --time 300000 --dt 100 --out-folder ${WRKDIR}/22-7-tuloksia/perfect/l-256