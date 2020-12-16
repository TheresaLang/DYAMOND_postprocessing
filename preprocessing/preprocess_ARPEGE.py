#!/usr/bin/env python
#SBATCH --job-name=ARPEGE_preprocessing
#SBATCH --output=logs/ARPEGE_preprocessing-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to preprocess raw IFS data (all steps before horizontal interpolation)
# Script has to be submitted with option --array=0-<Number_of_inputfiles> (one input file per
# time step)
import os
import numpy as np
from os.path import join
import preprocessing_tools as ptools
import filelists

# load configuration
config = ptools.config()

ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to infile
preprocess_dir = '/mnt/lustre02/work/um0878/users/tlang/work/dyamond/processing/preprocessing_ARPEGE'
# get list of input and output files and options for horizontal interpolation
infiles_2D, infiles_3D, outfileprefix, merge_list_3D, tempfile_list_3D, filelist_2D, tempfile_list_2D\
= filelists.get_preprocessing_ARPEGE_filelist(**config)

ptools.preprocess_ARPEGE_1(preprocess_dir, infiles_2D[ID], infiles_3D[ID], outfileprefix[ID], merge_list_3D[ID],\
                           tempfile_list_3D[ID], filelist_2D[ID], tempfile_list_2D[ID], **config)
