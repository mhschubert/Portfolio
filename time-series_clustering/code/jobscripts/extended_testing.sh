#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./../jobscripts/out/reeval.%j
#SBATCH -e ./../jobscripts/error/reeval.%j
# Initial working directory:
#SBATCH -D /ptmp/mschuber/ehs/r
# Job Name:
#SBATCH -J reev
#Partition
#SBATCH --partition=chubby
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=21
# Enable Hyperthreading:
##SBATCH --ntasks-per-core=2
# for OpenMP:
#SBATCH --cpus-per-task=1
#
#SBATCH --mem=368000
#SBATCH --mail-type=none
#SBATCH --mail-user=<userid>@coll.mpg.de
#
# Wall clock limit:
#SBATCH --time 24:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# For pinning threads correctly:
export OMP_PLACES=cores  

# Run the program:
module load r_anaconda
srun Rscript extended_configuration_testing.R
echo "job finished"
