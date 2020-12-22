"""
Ths is the main python script for running CoRAL analysis

Example command line usage: python3 Calc_CR_response_config.py /path/to/config_file

An example for the config_file is here: ./data/coral_serf.conf
An example for the input file list is here: ./data/mli_desc.list
An example for the file containing target coordinates is here: ./data/site_lat_lon_hgt_date1_date2_az_rg_initial.txt
Test run: python3 Calc_CR_response.py ./data/coral_serf.conf

required packages to be installed (on Gadi):
# pip install --user joblib
# pip install --user rasterio
"""
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
import sys, os.path
from coral import config as cf
from coral.dataio import read_input_files, write_radar_coords, plot_results
from coral.corner_reflector import loop


# check if config-file has been given as an argument of the main script call
print('')
if len(sys.argv) != 2:
    print('Exiting: Provide path to config-file as command line argument')
    print('')
    print('Usage: python3 Calc_CR_response.py <config-file>')
    exit()
else:
    config_file = sys.argv[1]
    print(f'Looking for CoRAL input data defined in {config_file}')

# read config-file and convert parameters to required data type
params = cf.get_config_params(config_file)

# General screen output
print(f'Results will be saved into {params[cf.OUT_DIR]}')
# Start the processing
print(' ')
print('Running CoRAL...')
print(' ')
# used to calculate runtime
run_start = datetime.now()

# check and read paths to input data
files_a, files_d, sites = read_input_files(params)


#######################################
# function for parallel loop processing
sitenames = sites.keys()
# note that dictionaries are unsorted and the variable name is hence not ordered
def processInput(name):

    cr = sites[name]

    if files_a:
        avgI_a, rcs_a, scr_a, clt_a, t_a, cr_new_a, cr_pos_a = loop(files_a, params[cf.SUB_IM], cr[0], params[cf.TARG_WIN_SZ], params[cf.CLT_WIN_SZ])
    if files_d:
        avgI_d, rcs_d, scr_d, clt_d, t_d, cr_new_d, cr_pos_d = loop(files_d, params[cf.SUB_IM], cr[1], params[cf.TARG_WIN_SZ], params[cf.CLT_WIN_SZ])

    if files_a and files_d:
        return name, cr_pos_a, avgI_a, rcs_a, scr_a, clt_a, t_a, cr_new_a, \
               cr_pos_d, avgI_d, rcs_d, scr_d, clt_d, t_d, cr_new_d
    elif files_a and not files_d:
        return name, cr_pos_a, avgI_a, rcs_a, scr_a, clt_a, t_a, cr_new_a
    elif files_d and not files_a:
        return name, cr_pos_d, avgI_d, rcs_d, scr_d, clt_d, t_d, cr_new_d


# parallel processing of MLI read and RCS, SCR calculation
num_cores = multiprocessing.cpu_count()
# num_cores results in 32 on the NCI which in turn results in an error
# hence the number of 16 cores is hard-coded here
results = Parallel(n_jobs=params[cf.N_JOBS])(delayed(processInput)(name) for name in sitenames)
#######################################


# extract results and plot images
print(' ') 
print('Creating output data...')
print(' ') 

# create output dir if it doesn't exist
if not os.path.exists(params[cf.OUT_DIR]):
    os.makedirs(params[cf.OUT_DIR])

plot_results(sites, results, params)


# write updated radar coordinates to a new file
print(' ')
if files_a:
    # new file with updated radar coordinates of CRs
    write_radar_coords(params[cf.ASC_CR_FILE_ORIG], params[cf.ASC_CR_FILE_NEW], sites, "asc")
if files_d:
    # new file with updated radar coordinates of CRs
    write_radar_coords(params[cf.DESC_CR_FILE_ORIG], params[cf.DESC_CR_FILE_NEW], sites, "desc")
        

# print out runtime
runtime = datetime.now() - run_start
print(' ')
print('Runtime: %s' % runtime)
