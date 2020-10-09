#!/usr/bin/env python
#SBATCH --job-name=net_radiation_calc
#SBATCH --output=logs/net_radiation_calc-%j.out

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
##SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to calculate net radiation fluxes from upward and downward fluxes
# One job corresponds to the processing of 
# Script has to be submitted with option --array=0-<Number_of_inputfiles>
import os
import numpy as np
from os.path import join
import preprocessing_tools as ptools
import filelists

# load configuration
config = ptools.config()
fluxes = ['SDTOA', 'SUTOA', 'STOA']
# get list of input and output files and options for horizontal interpolation
infiles, tempfiles, outfiles = filelists.get_netfluxcalculationfilelist(fluxes, **config)

ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to infile

ptools.calc_net_radiation_flux(infiles[ID], tempfiles[ID], outfiles[ID], fluxes)