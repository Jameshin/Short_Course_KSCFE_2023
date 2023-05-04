#!/bin/bash
#SBATCH -J horovod
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time 30:00:00
#SBATCH --gres=gpu:4
#SBATCH --comment pytorch #Application

module purge

module load python/3.7.1 gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 
source activate gs

horovodrun -np 4 python main.py 
