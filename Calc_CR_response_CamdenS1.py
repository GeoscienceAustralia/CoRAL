"""
Ths is an example python script for running CoRAL analysis

Example command line usage: python3 Calc_CR_response_config.py
"""
import glob
import numpy as np
from datetime import datetime
from coral.corner_reflector import loop
from coral.dataio import read_radar_coords, write_radar_coords
from coral.plot import *
from coral.plot2 import *
# for parallel processing: module joblib needs to be installed:
# e.g.: pip install --user joblib
from joblib import Parallel, delayed
import multiprocessing
import sys, os, os.path


#####################
### Config parameters
#####################
# data directories for ascending and/or descending data paths (use "-" if geometry doesn't exist)
data_path_asc = "-"
data_path_desc = "/g/data/dg9/INSAR_ANALYSIS/CAMDEN/S1/GAMMA/T147D"

# convention for for filenames of intensity images within the given data paths
file_name_conv = "/SLC/20*/r*8rlks.mli"

# name of file containing initial radar coordinates of CRs (located in the asc/desc data_path directory)
cr_file_name = "/CR_site_lon_lat_hgt_date1_date2_az_rg_initial.txt"
# name of file to write updated radar coordinates of CRs (located in the asc/desc data_path directory)
cr_file_name_new = "/CR_site_lon_lat_hgt_date1_date2_az_rg.txt"

# output directory
path_out = "/g/data/dg9/INSAR_ANALYSIS/CAMDEN/S1/GAMMA/CoRAL_output"

# target window
targ_win_sz = 5 # for multi-looked data (factor 2)
# targ_win_sz = 7 # for full-resolution data
# clutter window
clt_win_sz = 11
# clt_win_sz = 15 # for full-resolution data
# give cropped image size (for plotting)
sub_im = 51
#####################
### Config parameters
#####################


# Start the processing
print(' ')
print('Running CoRAL...')
print(' ')
# used to calculate runtime
run_start = datetime.now()

# if - is given as datapath, this geometry is not used
if data_path_asc == "-":
    asc_track_files = ""
else:
    asc_track_files = data_path_asc + file_name_conv
if data_path_desc == "-":
    desc_track_files = ""
else:
    desc_track_files =  data_path_desc + file_name_conv


# Get list of image files in asc/desc directories
if asc_track_files:
    print('Ascending data set given...')
    files_a = []
    for file in glob.glob(asc_track_files):
        files_a.append(file)
    files_a.sort()
    # read initial radar coordinates of CRs
    filename = data_path_asc + cr_file_name
    sites_a, az_a, rg_a = read_radar_coords(filename)  
if desc_track_files:
    print('Descending data set given...')
    files_d = []
    for file in glob.glob(desc_track_files):
        files_d.append(file)
    files_d.sort()
# read initial radar coordinates of CRs
    filename = data_path_desc + cr_file_name
    sites_d, az_d, rg_d = read_radar_coords(filename)  


# check if CR files have the same length
if asc_track_files and desc_track_files:
    if sites_a == sites_d:
        print("CR files for asc and desc tracks contain same set of sites.")
        keys = sites_a
        values = (np.array([[rg_a[i], az_a[i]],[rg_d[i], az_d[i]]], dtype=int) \
                 for i in range(0, len(sites_a)))
    else:
        print("ERROR: different CR sites given for asc and desc track.")
        sys.exit()
    print(' ')    
elif asc_track_files and not desc_track_files:
    keys = sites_a
    values = (np.array([[rg_a[i], az_a[i]],[-999, -999]], dtype=int) \
             for i in range(0, len(sites_a)))
elif desc_track_files and not asc_track_files:
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

    if asc_track_files:
        avgI_a, rcs_a, scr_a, clt_a, t_a, cr_new_a, cr_pos_a = loop(files_a, sub_im, cr[0], targ_win_sz, clt_win_sz)
    if desc_track_files:
        avgI_d, rcs_d, scr_d, clt_d, t_d, cr_new_d, cr_pos_d = loop(files_d, sub_im, cr[1], targ_win_sz, clt_win_sz)

    if asc_track_files and desc_track_files:
        return name, cr_pos_a, avgI_a, rcs_a, scr_a, clt_a, t_a, cr_new_a, \
               cr_pos_d, avgI_d, rcs_d, scr_d, clt_d, t_d, cr_new_d
    elif asc_track_files and not desc_track_files:
        return name, cr_pos_a, avgI_a, rcs_a, scr_a, clt_a, t_a, cr_new_a
    elif desc_track_files and not asc_track_files:
        return name, cr_pos_d, avgI_d, rcs_d, scr_d, clt_d, t_d, cr_new_d


# parallel processing of MLI read and RCS, SCR calculation
num_cores = multiprocessing.cpu_count()
# num_cores results in 32 on the NCI which in turn results in an error
# hence the number of 16 cores is hard-coded here
results = Parallel(n_jobs=16)(delayed(processInput)(name) for name in names)


# extract results and plot images

# create output dir if it doesn't exist
if not os.path.exists(path_out):
    os.makedirs(path_out)

print(' ') 
print('Creating output data...')
print(' ') 
for i in range(0, len(names)):
    
    # ascending and descending CRs in one plot
    if asc_track_files and desc_track_files:
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
        plot_mean_intensity2(avgI_a, avgI_d, cr_pos_a, cr_pos_d, targ_win_sz, clt_win_sz, name, path_out)
         
        # extract start and end date
        start_time = min(t_a[0], t_d[0])
        end_time = max(t_a[-1], t_d[-1])
        margin = (end_time - start_time)/50
        start = start_time - margin
        end = end_time + margin
        
        # Plot average clutter time series
        plot_clutter2(t_a, t_d, clt_a, clt_d, start, end, name, path_out) 
        # Plot scr time series
        plot_scr2(t_a, t_d, scr_a, scr_d, start, end, name, path_out) 
        # Plot rcs time series
        plot_rcs2(t_a, t_d, rcs_a, rcs_d, start, end, name, path_out)   
        
        # write RCS data to file
        filename_out = path_out + "/rcs_values_" + name + "_" + "Ascending.txt"
        fout = open(filename_out,'w')
        for time, value in zip(t_a,rcs_a):
            timestr = time.strftime("%Y%m%d")
            fout.write("%s %f\n" % (timestr, value))
        fout.close()
        filename_out = path_out + "/rcs_values_" + name + "_" + "Descending.txt"
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
        if asc_track_files:
           geom='Ascending'
           cr = np.array([cr_new, [-999, -999]])
        if desc_track_files:
           geom='Descending'
           cr = np.array([[-999, -999], cr_new])
        sites[name] = cr


        # Visualisation
        print('Site %s' % name)
        # Plot mean intensity image
        plot_mean_intensity(avgI, cr_pos, targ_win_sz, clt_win_sz, name, path_out)
        
        # extract start and end date
        start_time = t[0]
        end_time = t[-1]
        margin = (end_time - start_time)/50
        start = start_time - margin
        end = end_time + margin
        
        # Plot average clutter time series
        plot_clutter(t, clt, start, end, name, geom, path_out) 
        # Plot scr time series
        plot_scr(t, scr, start, end, name, geom, path_out) 
        # Plot rcs time series
        plot_rcs(t, rcs, start, end, name, geom, path_out) 
        # Plot RCS_SCR time series
        plot_rcs_scr(t, rcs, scr, start, end, name, path_out)   

        
        # write RCS data to file
        filename_out = path_out + "/rcs_values_" + name + "_" + geom + ".txt"
        fout = open(filename_out,'w')
        for time, value in zip(t,rcs):
            timestr = time.strftime("%Y%m%d")
            fout.write("%s %f\n" % (timestr, value))
        fout.close()  


# write updated radar coordinates to a new file
print(' ')
if asc_track_files:
    # new file with updated radar coordinates of CRs
    filename_init = data_path_asc + cr_file_name
    filename = data_path_asc + cr_file_name_new
    write_radar_coords(filename_init, filename, sites, "asc") 
if desc_track_files:
    # new file with updated radar coordinates of CRs
    filename_init = data_path_desc + cr_file_name
    filename = data_path_desc + cr_file_name_new
    write_radar_coords(filename_init, filename, sites, "desc") 
        

# print out runtime
runtime = datetime.now() - run_start
print(' ')
print('Runtime: %s' % runtime)
