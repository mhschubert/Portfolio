#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./../jobscripts/gridsearch.%j
#SBATCH -e ./../jobscripts/gridsearch.%j
# Initial working directory:
#SBATCH -D /ptmp/mschuber/technical/commasoft/tech_1/code
# Job Name:
#SBATCH -J gridsearch
#Partition
#SBATCH --partition=medium
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40

##SBATCH --mem=185000
#SBATCH --mail-type=none
#SBATCH --mail-user=schubert@coll.mpg.de
#
# Wall clock limit:
#SBATCH --time 24:00:00
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores

# Run the program:
modul python technical_comma_1.py
#R CMD BATCH --no-save --no-restore main.R TRUE 8 1

