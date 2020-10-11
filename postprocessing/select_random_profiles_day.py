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
# separately for each day

import os
import postprocessing_tools as ptools

ID = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) # ID corresponds to model
timesteps = np.arange(8 * ID, 8 * (ID + 1)) # timesteps corresponding to one day

# load configuration
config = ptools.config()
# add timesteps and filename suffix to config
config['timesteps'] = timesteps
config['filename_suffix'] = ID

model = config['models'][0]
run = config['runs'][0]

ptools.select_random_profiles_new(model, run, **config)