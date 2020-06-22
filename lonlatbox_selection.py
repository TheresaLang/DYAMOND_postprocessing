#!/usr/bin/env python
#SBATCH --job-name=lonlatbox_selection
#SBATCH --output=logs/lonlatbox_selection-%j.out
#SBATCH --account=mh1126         # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48       # Specify number of CPUs per task
#SBATCH --time=08:00:00          # Set a limit on the total run time
#SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
##SBATCH --mem=0                  # All mem on node
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL        # Notify user by email in case of job failure

# Script to average over all timesteps contained on one (merged) file.
# Model, run and time period have to be specified in config.json.
# One job corresponds to the processing of one variable.
# Script has to be submitted with option --array=0-<Number_of_variables>
import os
import numpy as np
from os.path import join
import processing_tools as ptools
import filelists

# load config
config = ptools.config()
# get list of filenames for input and output files (one entry per variable)
infiles, outfiles = filelists.get_sellonlatboxfilelist(**config)

# ID of this job
# Each job gets an own ID from 0 to N, where N is the number of jobs in the job array
# N is specified when job is submitted with 
# $ sbatch --array=0-N time_averaging.py
ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to variable

# time averaging for the file with index ID in the filelist
# here, the index corresponds to the variable
ptools.select_lonlatbox(infiles[ID], outfiles[ID], **config)