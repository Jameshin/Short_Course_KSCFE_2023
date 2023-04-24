#!/bin/bash
#SBATCH -J horovod
#SBATCH -p cas_v100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=2
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time 30:00:00
#SBATCH --gres=gpu:8
#SBATCH --comment pytorch #Application

module purge
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 cmake/3.16.9
source activate gs

horovodrun -np 8 python main.py 
