import numpy as np
from os.path import join
import processing_tools as ptools

model = 'NICAM'
run = '3.5km'
variables = ['TEMP', 'QV', 'PRES']
grid_res = 0.1
time_period = ['2016-08-11', '2016-09-10']
filename = '/work/mh1126/m300773/DYAMOND/{}/{}-{}_{}_0.10deg{}.nc'

# horizontal interpolation and merging to one file
for var in variables:
    print('Horizontal interpolation of {}'.format(var))
    outfile = filename.format(model, model, run, var, '')
    ptools.horizontal_interpolation(model, run, var, grid_res, time_period, outfile, temp_dir='/scratch/m/m300773/')

# temporal averaging
#for var in variables:
#    print('Temporal averaging of {}'.format(var))
#    infile = filename.format(model, model, run, var, '')
#    outfile = filename.format(model, model, run, var, '_timemean')
#    ptools.temporal_averaging(infile, outfile, overwrite=False)

# make files look like ICON files

# Vertical interpolation
#for var in variables:
#    print('Vertical interpolation of {}'.format(var))
#    infile = filename.format(model, model, run, var, '_timemean')
#    outfile = filename.format(model, model, run, var, '_timemean_vertint')
#    ptools.vertical_interpolation(infile, outfile, model, var, overwrite=False)
