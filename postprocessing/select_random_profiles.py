#!/usr/bin/env python
#SBATCH --job-name=random_profile_selection
#SBATCH --output=logs/random_profile_selection-%j.out
##SBATCH --error=random_profile_selection-%j.err

#SBATCH --account=mh1126       # Charge resources on this project account
#SBATCH --partition=compute,compute2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36     # Specify number of CPUs per task
#SBATCH --time=08:00:00        # Set a limit on the total run time
#SBATCH --mem=0                # use all memory on node
#SBATCH --constraint=256G      # only run on fat memory nodes (needed for NICAM)
#SBATCH --monitoring=meminfo=10,cpu=5,lustre=5
##SBATCH --mail-type=FAIL      # Notify user by email in case of job failure

# Script to randomly select profiles from horizontally interpolated DYAMOND data
import os
import postprocessing_tools as ptools

ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to model

# load configuration
config = ptools.config()
# add timesteps and filename suffix to config
config['timesteps'] = None
config['filename_suffix'] = ''

models = config['models']
runs = config['runs']

ptools.select_random_profiles(model=models[ID], run=runs[ID], **config)


# import os
# import numpy as np
# from os.path import join
# import postprocessing_tools as ptools
# import filelists

# reload(ptools)
# # load configuration
# config = ptools.config()

# ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to model
# num_samples = config['num_samples']
# models, runs, infiles, outfiles = filelists.get_samplefilelist(num_samples, **config)
# heightfile = join(config['data_dir'], models[ID], 'target_height.nc')
# landmaskfile = '/mnt/lustre02/work/mh1126/m300773/DYAMOND/ICON/land_mask.nc'

# ptools.select_random_profiles(models[ID], runs[ID], num_samples, infiles[ID], outfiles[ID], heightfile, landmaskfile, **config)