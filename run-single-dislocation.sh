#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --job-name=partial-mini
#SBATCH --mem-per-cpu=5G
#SBATCH --output=2025-05-23-miniajo-partial.out
#SBATCH --array=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi


module load scicomp-python-env

srun python3 main.py -p 1 -dt 0.1 -t 1000000 -f results/3-6-2025-dislokaatio single -R 0.001 -m 0.0