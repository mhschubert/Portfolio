#!/bin/bash -l

# This is a jobscript for a SLURM scheduler running on an HPC cluster
# Standard output and error:
#SBATCH -o ./../jobscripts/out/find_config_MULTI_grid_out_all.%j
#SBATCH -e ./../jobscripts/error/find_config_MULTI_grid_err_all.%j
# Initial working directory:
#SBATCH -D /ptmp/USERNAME/ehs/r
# Job Name:
#SBATCH -J gr_all
#Partition
#SBATCH --partition=fat
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
# Enable Hyperthreading:
#SBATCH --ntasks-per-core=2
# for OpenMP:
#SBATCH --cpus-per-task=1
#
##SBATCH --mem=42000
#SBATCH --mail-type=none
#SBATCH --mail-user=<userid>@coll.mpg.de
#
# Wall clock limit:
#SBATCH --time 24:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# For pinning threads correctly:
export OMP_PLACES=threads
export SLURM_HINT=multithread 

# Run the program:
module load r_anaconda
srun Rscript find_cnfg_grid_internal_task_array.R TRUE 1 1

echo "job finished"
