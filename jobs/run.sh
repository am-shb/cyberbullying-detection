#!/bin/bash
#SBATCH --account=***
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00-4:00
#SBATCH --output=slurm/%N-%j.out
#SBATCH --mail-user=***
#SBATCH --mail-type=ALL


module load python/3.8 scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index sklearn xgboost imblearn

python src/binary.py
