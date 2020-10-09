#!/usr/bin/env python
#SBATCH --job-name=preprocessing
#SBATCH --output=logs/preprocessing-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to preprocess raw 3D DYAMOND output to make it usable for horizontal
# interpolation with CDOs.
# Model, run and time period have to be specified before in config.json.
# One job corresponds to the processing of one output file.
# Script has to be submitted with option --array=0-<Number_of_inputfiles>
import os
import numpy as np
from os.path import join
import preprocessing_tools as ptools
import filelists

# load configuration
config = ptools.config()

# get list of input and output files and options for horizontal interpolation
models, infiles, tempfiles, outfiles, option_1, option_2 = filelists.get_preprocessingfilelist(**config)

ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to infile

# perform preprocessing steps
ptools.preprocess(models[ID], infiles[ID], tempfiles[ID], outfiles[ID], option_1[ID], option_2[ID], **config)