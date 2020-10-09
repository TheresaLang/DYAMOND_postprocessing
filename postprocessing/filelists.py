def get_samplefilelist(num_samples_tot, models, runs, variables, time_period, lonlatbox, data_dir, sample_days, experiment=None, day=None, **kwargs):
    """ Returns filelist needed to perform Monte Carlo Sampling of Profiles.
    
    Parameters:
        num_samples_tot (num): number of samples to draw
        models (list of str): names of models
        runs (list of str): names of model runs
        variables (list of str): list of variable names
        time_period (list of str): list containing start and end of time period as string 
            in the format YYYY-mm-dd 
        lonlatbox (list of ints): borders of lon-lat box to sample from [lonmin, lonmax, latmin, latmax]
        data_dir (str): path to directory of input file
        experiment (num): experiment number (if several experiments are performed), needed for output filename
    """  
    #model = models[0]
    #run = runs[0]
    variables_2D = ['OLR', 'OLRC', 'STOA', 'STOAC', 'IWV', 'CRH', 'SURF_PRES', 'SST']
    time = pd.date_range(time_period[0], time_period[1], freq='1D')
    start_date_in = time[0].strftime("%m%d")
    end_date_in = time[-1].strftime("%m%d")
    if day is None:
        start_date_out = start_date_in
        end_date_out = end_date_in
    else:
        start_date_out = time[day].strftime("%m%d")
        end_date_out = time[day].strftime("%m%d")
        
    if lonlatbox[:2] == [-180, 180]:
        latlonstr = ''
    else:
        latlonstr = f'_{lonlatbox[0]}_{lonlatbox[1]}_{lonlatbox[2]}_{lonlatbox[3]}'
    if experiment is not None:
        expstr = f'_{str(experiment)}'
    else:
        expstr = ''
    if sample_days == 'all':
        sample_day_str = ''
    else:
        sample_day_str = '_'+sample_days
    
    model_list = []
    run_list = []
    infile_list = []
    outfile_list = []
    
    for model, run in zip(models, runs):
        model_list.append(model)
        run_list.append(run)
        infile_sublist = []
        outfile_sublist = []
        for var in variables:
            if model in ['ICON', 'MPAS'] and var == 'W':
                var = 'WHL'
            if var == 'SST':
                file = f'Initialization/DYAMOND_SST_{start_date_in}-{end_date_in}_hinterp_merged.nc'
            elif model in ['IFS', 'GEOS', 'FV3', 'ARPEGE', 'ERA5'] and var not in variables_2D:
                file = f'{model}/{model}-{run}_{var}_hinterp_vinterp_merged_{start_date_in}-{end_date_in}.nc'
            else: 
                file = f'{model}/{model}-{run}_{var}_hinterp_merged_{start_date_in}-{end_date_in}.nc'
            file = os.path.join(data_dir, file)
            infile_sublist.append(file) 

        for var in variables + ['IWV', 'CRH', 'lon', 'lat', 'ind_lon', 'ind_lat', 'sort_ind']:
            outfile = f'{model}-{run}_{var}_sample_{num_samples_tot}_{start_date_out}-{end_date_out}{sample_day_str}{latlonstr}{expstr}.nc'
            outfile = os.path.join(data_dir, model, 'random_samples', outfile)
            outfile_sublist.append(outfile)
        infile_list.append(infile_sublist)
        outfile_list.append(outfile_sublist)
    
    return model_list, run_list, infile_list, outfile_list