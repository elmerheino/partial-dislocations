#!/bin/bash
#SBATCH --time={hours}:{minutes}:{seconds}
#SBATCH --job-name={job-name}
#SBATCH --mem-per-cpu=200M
#SBATCH --output={job-name}.out
#SBATCH --cpus-per-task={cores}
#SBATCH --array={arr-start}-{arr-end}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

srun python3 ../depinningFromRel.py --folder {input-path} {perfect-partial} --points 30 --cores {cores} --task-id ${SLURM_ARRAY_TASK_ID} --time {rel-time} --dt {sample-dt} --out-folder {output-path}