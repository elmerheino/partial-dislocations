#!/bin/bash
#SBATCH --time={hours}:00:00
#SBATCH --job-name={name}
#SBATCH --mem-per-cpu=200M
#SBATCH --output={name}.out
#SBATCH --cpus-per-task={cores}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi

module load scicomp-python-env

TIME={time}
SYSTEM={system}
  
srun python3 ../initialRelaxations.py new --rmin {rmin} --rmax {rmax} --rpoints {rpoints} --seeds {seeds} --n ${SYSTEM} --length ${SYSTEM} --folder {folder} -c {cores} --d0 {d0} {perfect_partial}