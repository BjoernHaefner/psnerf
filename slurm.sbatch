#!/bin/bash
#SBATCH --job-name="psnerf"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=24G
#SBATCH --time=01-00:00:00 ##DD-HH:MM:SS
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/haefner/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/haefner/slurm-%j.out
#SBATCH --partition=DEADLINE
##SBATCH --exclude=node[2-6] ##exclude 12gb GPUs
#SBATCH --exclude=node[2-6,8,10] ##exclude 12gb and 16gb GPUs
##SBATCH --exclude=node[1-10] ##exclude less than 48gb GPUs
##SBATCH --exclude=node[1-10,15-18] ##exclude less than 48gb GPUs and a40
##SBATCH --exclude=node[1-10,14,15-18] ##exclude less than 48gb GPUs and a40 and rtx a6000
##SBATCH --exclude=node[1-10,11-13,14,15-18] ##exclude less than 48gb GPUs and a40 and rtx a6000 and rtx 8000
##SBATCH --exclude=node[1-15,17-18] ##exclude less than 48gb GPUs and a40 and rtx a6000 and rtx 8000
#SBATCH --comment="Our data"

## call to find all data sets:
## for f in $(find dataset -maxdepth 1 -type d -exec basename {} \; | grep -E 'real|synthetic'); do sbatch slurm.sbatch $f; done
echo "srun python quick_start.py -n ${1}"
srun python quick_start.py -n $1

