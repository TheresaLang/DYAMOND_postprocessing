#!/usr/bin/env python
#SBATCH --job-name=model_mergetime
#SBATCH --output=logs/model_mergetime-%j.out

#SBATCH --account=mh1126         # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48       # Specify number of CPUs per task
#SBATCH --time=08:00:00          # Set a limit on the total run time
#SBATCH --mem=0                  # use all memory on node
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL        # Notify user by email in case of job failure

# Script to merge several files containing one timestep each into one file
# containing all timesteps.
# Model, run and time period have to be specified in config.json.
# One job corresponds to the processing of one variable.
# Script has to be submitted with option --array=0-<Number_of_models*Number_of_variables>
import os
import numpy as np
from os.path import join
import preprocessing_tools as ptools
import filelists

# load config
config = ptools.config()

# get lists with input files (that shall be merged) and output files 
infiles, outfiles = filelists.get_mergingfilelist(**config)

ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to variable

# perform time merging
ptools.merge_timesteps(infiles[ID], outfiles[ID], **config)

