#!/bin/bash
#SBATCH --time={hours}:{minutes}:{seconds}
#SBATCH --job-name={job-name}
#SBATCH --mem-per-cpu=200M
#SBATCH --output={job-name}.out
#SBATCH --cpus-per-task={cores}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env


srun python3 ../depinningFromRel.py continue --initial-relaxations {initial-relaxations} --depinning-params {depinning-params} --task-id {task-id}