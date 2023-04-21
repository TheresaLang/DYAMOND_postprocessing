#!/usr/bin/env python
#SBATCH --job-name=rh_calc
#SBATCH --output=logs/rh_calc-%j.out
##SBATCH --error=rh_calc-%j.err

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to calculate relative humidity using the ncl function "relhum"
import os
import numpy as np
from os.path import join
from netCDF4 import Dataset
import preprocessing_tools as ptools
import filelists
# ID of this job
# Each job gets an own ID from 0 to N, where N is the number of jobs in the job array
# N is specified when job is submitted with 
# $ sbatch --array=0-N ncl_relative_humidity_calculation.py
ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to infile

# load configuration
config = ptools.config()

# get list of input and output files
qfiles, tempfiles, outfiles = filelists.get_nclrhcalculation_filelist(**config)

ptools.calc_relative_humidity_ncl(qfiles[ID], tempfiles[ID], outfiles[ID])