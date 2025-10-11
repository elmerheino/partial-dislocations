#!/bin/bash
#SBATCH --job-name=fire_critical_force
#SBATCH --output=fire_critical_force_%A_%a.out
#SBATCH --error=fire_critical_force_%A_%a.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elmer.heino@aalto.fi
#SBATCH --array=1-10

# Load modules (edit as needed for your environment)
module load scicomp-python-env


N=$1
L=$1
D0=$3
CORES=10
FOLDER_NAME=$2                  # Unused variable, completely irrelevant, still must be specified.
SEED=${SLURM_ARRAY_TASK_ID}
TIME=1000.0
DT=0.01
POINTS=15
RMIN=-4.0
RMAX=0
SAVE_FOLDER=${FOLDER_NAME}      # Save folder is where all the data is saved
TAUPOINTS=100
# Run the script
srun python3 ../criticalForceUsingFIRE.py \
    --N $N \
    --L $L \
    --cores $CORES \
    --folder_name ${FOLDER_NAME}/backups \
    --d0 $D0 \
    --seed $SEED \
    --time $TIME \
    --dt $DT \
    --points $POINTS \
    --rmin $RMIN \
    --rmax $RMAX \
    --save_folder $SAVE_FOLDER \
    --taupoints $TAUPOINTS \
    --partial
