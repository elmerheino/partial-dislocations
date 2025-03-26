#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=depinning.out
#SBATCH --array=0-10000
#SBATCH --cpus-per-task=9

DATE=$(date +"%Y-%m-%d")
echo $DATE

CORES=9

# You may put the commands below:
module load scicomp-python-env

NOISE_POINTS=100
NOISE_MIN=0.01
NOISE_MAX=100

SEED=$((${SLURM_ARRAY_TASK_ID} / ${NOISE_POINTS}))
NOISE=$((${SLURM_ARRAY_TASK_ID} % ${NOISE_POINTS}))
echo "SEED: ${SEED} NOISE: ${NOISE}"

# Seed count is array-max/noise points

# Job step
srun python3 main.py --seed ${SEED} -R ${NOISE} -rmin ${NOISE_MIN} -rmax ${NOISE_MAX} -rpoints ${NOISE_POINTS} -f ${WRKDIR}/results/${DATE}/partial-dislocation -tmin 2.65 -tmax 3.0 -p 100 -dt 0.05 -c ${CORES} -t 10000 --partial
srun python3 main.py --seed ${SEED} -R ${NOISE} -rmin ${NOISE_MIN} -rmax ${NOISE_MAX} -rpoints ${NOISE_POINTS}  -f ${WRKDIR}/results/${DATE}/single-dislocation -tmin 2.65 -tmax 3.0 -p 100 -dt 0.05 -c ${CORES} -t 10000 --single