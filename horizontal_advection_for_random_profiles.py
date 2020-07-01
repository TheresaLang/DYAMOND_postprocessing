#!/usr/bin/env python
#SBATCH --job-name=horizontal_advection
#SBATCH --output=logs/horizontal_advection-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
#SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to calculate horizontal advection of moisture for randomly selected profiles

import os
import processing_tools as ptools

# load configuration
config = ptools.config()
models = config['models']
runs = config['runs']
ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to model

ptools.advection_for_random_profiles(models[ID], runs[ID], **config)