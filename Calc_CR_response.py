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
import numpy as np
from datetime import datetime
from coral.corner_reflector import loop
from coral.dataio import read_radar_coords, write_radar_coords
from coral.plot import *
from coral.plot2 import *
from joblib import Parallel, delayed
import multiprocessing
import sys, os, os.path
from coral import config as cf
from configparser import ConfigParser


class Configuration:
    def __init__(self, config_file_path):
        parser = ConfigParser()
        parser.optionxform = str
        # mimic header to fulfil the requirement for configparser
        with open(config_file_path) as stream:
            parser.read_string("[root]\n" + stream.read())

        for key, value in parser["root"].items():
            self.__dict__[key] = value

def _params_from_conf(config_file):
    config_file = os.path.abspath(config_file)
    config = Configuration(config_file)
    params = config.__dict__
    return params


print('')
if len(sys.argv) != 2:
    print('Exiting: Provide path to config-file as command line argument')
    print('')
    print('Usage: python3 Calc_CR_response.py <config-file>')
    exit()
else:
    config_file = sys.argv[1]
    print(f'Looking for CoRAL input data defined in {config_file}')


params = _params_from_conf(config_file)

print(f'Results will be saved into {params[cf.OUT_DIR]}')
# Start the processing
print(' ')
print('Running CoRAL...')
print(' ')
# used to calculate runtime
run_start = datetime.now()


# if no datapath is given in the config file, this geometry is not used
if params[cf.ASC_LIST]:
    print(f'Reading data from file list {params[cf.ASC_LIST]}')
    # read the paths to ascending-pass backscatter images
    with open(params[cf.ASC_LIST]) as f_in:
        files_a = [line.rstrip() for line in f_in]
        files_a.sort()
    # read the ascending-pass radar coordinates of targets
    filename = params[cf.ASC_CR_FILE_ORIG]
    sites_a, az_a, rg_a = read_radar_coords(filename)
else:
    print('No ascending-pass file list supplied')
    # read asc pass data files
if params[cf.DESC_LIST]:
    print(f'Reading data from file list {params[cf.DESC_LIST]}')
    # read the paths to descending-pass backscatter images
    with open(params[cf.DESC_LIST]) as f_in:
        files_d = [line.rstrip() for line in f_in]
        files_d.sort()
    # read the descending-pass radar coordinates of targets
    filename = params[cf.DESC_CR_FILE_ORIG]
    sites_d, az_d, rg_d = read_radar_coords(filename)
else:
    print('No descending-pass file list supplied')


# check if CR files have the same length
if params[cf.ASC_LIST] and params[cf.DESC_LIST]:
    if sites_a == sites_d:
        print("CR files for asc and desc tracks contain same set of sites.")
        keys = sites_a
        values = (np.array([[rg_a[i], az_a[i]],[rg_d[i], az_d[i]]], dtype=int) \
                 for i in range(0, len(sites_a)))
    else:
        print("ERROR: different CR sites given for asc and desc track.")
        sys.exit()
    print(' ')    
elif params[cf.ASC_LIST] and not params[cf.DESC_LIST]:
    keys = sites_a
    values = (np.array([[rg_a[i], az_a[i]],[-999, -999]], dtype=int) \
             for i in range(0, len(sites_a)))
elif params[cf.DESC_LIST] and not params[cf.ASC_LIST]:
    keys = sites_d
    values = (np.array([[-999, -999],[rg_d[i], az_d[i]]], dtype=int) \
             for i in range(0, len(sites_d)))

sites = dict(zip(keys, values))


##################################
# function for parallel loop processing
names = sites.keys()
# note that dictionaries are unsorted and the variable name is hence not ordered
def processInput(name):

    cr = sites[name]

    if params[cf.ASC_LIST]:
        avgI_a, rcs_a, scr_a, clt_a, t_a, cr_new_a, cr_pos_a = loop(files_a, int(params[cf.SUB_IM]), cr[0], int(params[cf.TARG_WIN_SZ]), int(params[cf.CLT_WIN_SZ]))
    if params[cf.DESC_LIST]:
        avgI_d, rcs_d, scr_d, clt_d, t_d, cr_new_d, cr_pos_d = loop(files_d, int(params[cf.SUB_IM]), cr[1], int(params[cf.TARG_WIN_SZ]), int(params[cf.CLT_WIN_SZ]))

    if params[cf.ASC_LIST] and params[cf.DESC_LIST]:
        return name, cr_pos_a, avgI_a, rcs_a, scr_a, clt_a, t_a, cr_new_a, \
               cr_pos_d, avgI_d, rcs_d, scr_d, clt_d, t_d, cr_new_d
    elif params[cf.ASC_LIST] and not params[cf.DESC_LIST]:
        return name, cr_pos_a, avgI_a, rcs_a, scr_a, clt_a, t_a, cr_new_a
    elif params[cf.DESC_LIST] and not params[cf.ASC_LIST]:
        return name, cr_pos_d, avgI_d, rcs_d, scr_d, clt_d, t_d, cr_new_d


# parallel processing of MLI read and RCS, SCR calculation
num_cores = multiprocessing.cpu_count()
# num_cores results in 32 on the NCI which in turn results in an error
# hence the number of 16 cores is hard-coded here
results = Parallel(n_jobs=16)(delayed(processInput)(name) for name in names)


# extract results and plot images

# create output dir if it doesn't exist
if not os.path.exists(params[cf.OUT_DIR]):
    os.makedirs(params[cf.OUT_DIR])

print(' ') 
print('Creating output data...')
print(' ') 
for i in range(0, len(names)):
    
    # ascending and descending CRs in one plot
    if params[cf.ASC_LIST] and params[cf.DESC_LIST]:
        # read result arrays of parallel function, both geometries
        name = results[i][0]
        cr_pos_a = results[i][1]
        avgI_a = results[i][2]
        rcs_a = results[i][3]
        scr_a = results[i][4]
        clt_a = results[i][5]
        t_a = results[i][6]
        cr_new_a = results[i][7]
        cr_pos_d = results[i][8]
        avgI_d = results[i][9]
        rcs_d = results[i][10]
        scr_d = results[i][11]
        clt_d = results[i][12]
        t_d = results[i][13]
        cr_new_d = results[i][14]
        
        # add new coords to sites dictionary
        cr = np.array([cr_new_a, cr_new_d])
        sites[name] = cr
        
        # Visualisation
        print('Site %s' % name)
        # Plot mean intensity image
        plot_mean_intensity2(avgI_a, avgI_d, cr_pos_a, cr_pos_d, int(params[cf.TARG_WIN_SZ]), int(params[cf.CLT_WIN_SZ]), name, params[cf.OUT_DIR])
         
        # extract start and end date
        start_time = min(t_a[0], t_d[0])
        end_time = max(t_a[-1], t_d[-1])
        margin = (end_time - start_time)/50
        start = start_time - margin
        end = end_time + margin
        
        # Plot average clutter time series
        plot_clutter2(t_a, t_d, clt_a, clt_d, start, end, name, params[cf.OUT_DIR])
        # Plot scr time series
        plot_scr2(t_a, t_d, scr_a, scr_d, start, end, name, params[cf.OUT_DIR])
        # Plot rcs time series
        plot_rcs2(t_a, t_d, rcs_a, rcs_d, start, end, name, params[cf.OUT_DIR])
        
        # write RCS data to file
        filename_out = params[cf.OUT_DIR] + "/rcs_values_" + name + "_" + "Ascending.txt"
        fout = open(filename_out,'w')
        for time, value in zip(t_a,rcs_a):
            timestr = time.strftime("%Y%m%d")
            fout.write("%s %f\n" % (timestr, value))
        fout.close()
        filename_out = params[cf.OUT_DIR] + "/rcs_values_" + name + "_" + "Descending.txt"
        fout = open(filename_out,'w')
        for time, value in zip(t_d,rcs_d):
            timestr = time.strftime("%Y%m%d")
            fout.write("%s %f\n" % (timestr, value))
        fout.close() 
         
    # one geometry (ascending or descending)    
    else:
        # read result arrays of parallel function, one geometry
        name = results[i][0]
        cr_pos = results[i][1]
        avgI = results[i][2]
        rcs = results[i][3]
        scr = results[i][4]
        clt = results[i][5]
        t = results[i][6]
        cr_new = results[i][7]

        # add new coords to sites dictionary
        if params[cf.ASC_LIST]:
           geom='Ascending'
           cr = np.array([cr_new, [-999, -999]])
        if params[cf.DESC_LIST]:
           geom='Descending'
           cr = np.array([[-999, -999], cr_new])
        sites[name] = cr


        # Visualisation
        print('Site %s' % name)
        # Plot mean intensity image
        plot_mean_intensity(avgI, cr_pos, int(params[cf.TARG_WIN_SZ]), int(params[cf.CLT_WIN_SZ]), name, params[cf.OUT_DIR])
        
        # extract start and end date
        start_time = t[0]
        end_time = t[-1]
        margin = (end_time - start_time)/50
        start = start_time - margin
        end = end_time + margin
        
        # Plot average clutter time series
        plot_clutter(t, clt, start, end, name, geom, params[cf.OUT_DIR])
        # Plot scr time series
        plot_scr(t, scr, start, end, name, geom, params[cf.OUT_DIR])
        # Plot rcs time series
        plot_rcs(t, rcs, start, end, name, geom, params[cf.OUT_DIR])
        # Plot RCS_SCR time series
        plot_rcs_scr(t, rcs, scr, start, end, name, params[cf.OUT_DIR])

        
        # write RCS data to file
        filename_out = params[cf.OUT_DIR] + "/rcs_values_" + name + "_" + geom + ".txt"
        fout = open(filename_out,'w')
        for time, value in zip(t,rcs):
            timestr = time.strftime("%Y%m%d")
            fout.write("%s %f\n" % (timestr, value))
        fout.close()  


# write updated radar coordinates to a new file
print(' ')
if params[cf.ASC_LIST]:
    # new file with updated radar coordinates of CRs
    write_radar_coords(params[cf.ASC_CR_FILE_ORIG], params[cf.ASC_CR_FILE_NEW], sites, "asc")
if params[cf.DESC_LIST]:
    # new file with updated radar coordinates of CRs
    write_radar_coords(params[cf.DESC_CR_FILE_ORIG], params[cf.DESC_CR_FILE_NEW], sites, "desc")
        

# print out runtime
runtime = datetime.now() - run_start
print(' ')
print('Runtime: %s' % runtime)
