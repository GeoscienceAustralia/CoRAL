"""
This python module contains functions for reading Synthetic Aperture 
Radar images
"""
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import sys, os
from coral import config as cf
from coral.plot import *
from coral.plot2 import *


def read_input_files(params):
    """
    Reads input files and performs checks on existence

    :param dict params: config parameters
    :return: list files_a: containing paths to asc backscatter files (none if not given)
    :return: list files_d: containing paths to desc backscatter files (none if not given)
    :return: dict sites: containing radar target sites and array with range and azimuth coordinates
                         (-999 is used for non-existent geometry if applicable)
    """
    # if no datapath is given in the config file, this geometry is not used
    # read asc pass data files
    if params[cf.ASC_LIST]:
        if not os.path.exists(params[cf.ASC_LIST]):
            raise Exception(f'{params[cf.ASC_LIST]} does not exist')
        else:
            print(f'Reading data from file list {params[cf.ASC_LIST]}')
            # read the paths to ascending-pass backscatter images
            with open(params[cf.ASC_LIST]) as f_in:
                files_a = [line.rstrip() for line in f_in]
                files_a.sort()
        # read the ascending-pass radar coordinates of targets
        if not os.path.exists(params[cf.ASC_CR_FILE_ORIG]):
            raise Exception(f'{params[cf.ASC_CR_FILE_ORIG]} does not exist')
        else:
            print('')
            sites_a, az_a, rg_a = read_radar_coords(filename)
    # set files variable to None if no asc file list supplied
    else:
        print('No ascending-pass file list supplied')
        files_a = None
    # read desc pass data files
    if params[cf.DESC_LIST]:
        if not os.path.exists(params[cf.DESC_LIST]):
            raise Exception(f'{params[cf.DESC_LIST]} does not exist')
        else:
            print(f'Reading data from file list {params[cf.DESC_LIST]}')
            # read the paths to descending-pass backscatter images
            with open(params[cf.DESC_LIST]) as f_in:
                files_d = [line.rstrip() for line in f_in]
                files_d.sort()
        # read the descending-pass radar coordinates of targets
        filename = params[cf.DESC_CR_FILE_ORIG]
        if not os.path.exists(params[cf.DESC_CR_FILE_ORIG]):
            raise Exception(f'{params[cf.DESC_CR_FILE_ORIG]} does not exist')
        else:
            print('')
            sites_d, az_d, rg_d = read_radar_coords(filename)
    # set files variable to None if no desc file list supplied
    else:
        print('No descending-pass file list supplied')
        files_d = None

    # check if CR files have the same length
    if files_a and files_d:
        if sites_a == sites_d:
            print("CR files for asc and desc tracks contain same set of sites.")
            keys = sites_a
            values = (np.array([[rg_a[i], az_a[i]], [rg_d[i], az_d[i]]], dtype=int) \
                      for i in range(0, len(sites_a)))
        else:
            print("ERROR: different CR sites given for asc and desc track.")
            sys.exit()
        print(' ')
    elif files_a and not files_d:
        keys = sites_a
        values = (np.array([[rg_a[i], az_a[i]], [-999, -999]], dtype=int) \
                  for i in range(0, len(sites_a)))
    elif files_d and not files_a:
        keys = sites_d
        values = (np.array([[-999, -999], [rg_d[i], az_d[i]]], dtype=int) \
                  for i in range(0, len(sites_d)))
    sites = dict(zip(keys, values))

    return files_a, files_d, sites


def readfile(file, sub_im, cr):
    """Function to read an image file"""

    root, ext = os.path.splitext(file)

    if ext == '.tif':
        print('Reading tiff image:', file)
        par = readpar(root + '.mli.par')
        data = readimg(file, sub_im, cr)

    elif ext == '.img':
        print('Reading ENVI image:', file)
        #par = readhdr(root + '.hdr')
        data = readimg(file, sub_im, cr)

    else: # must be GAMMA flat binary float format
        print('Reading flat binary image', file)
        par = readpar(root + ext + '.par')
        data = readmli(file, par, sub_im, cr)

    # extract relevant metadata
    rho_r = float(par['range_pixel_spacing'].split()[0])
    rho_a = float(par['azimuth_pixel_spacing'].split()[0])
    theta = float(par['incidence_angle'].split()[0])

    return data, rho_r, rho_a, theta


def readpar(file):
    """Function to read a GAMMA 'par' file into a dictionary"""
    par={}
    with open(file) as f:
        for line in f:
            if "Gamma" or " " in line:
                break # ignore header line
        for line in f:
            line=line.rstrip() # remove blank lines and whitespace
            if line and not "title" in line:
                (key, val) = line.split(":")
                par[str(key)] = val
    return par


def readmli(datafile, par, sub_im, cr):
    """Function to read a GAMMA mli file and provide a subsetted image"""
    ct = int(par['range_samples']) * int(par['azimuth_lines'])

    dt = np.dtype('>f4') # GAMMA files are big endian 32 bit float

    d = np.fromfile(datafile, dtype=dt, count=ct)

    d = d.reshape(int(par['azimuth_lines']), int(par['range_samples']))
    #print("Number of elements and size of the array is",d.size, d.shape)
    #d[d==0]= np.nan # convert zeros to nan
    return d[cr[1]-sub_im:cr[1]+sub_im,cr[0]-sub_im:cr[0]+sub_im]


def readimg(datafile, sub_im, cr):
    """Function to read a file and provide a subsetted image"""

    with rio.open(datafile) as src:
        d = src.read(1, window=Window(cr[0]-sub_im, cr[1]-sub_im, sub_im*2, sub_im*2))

    #print("Number of elements and size of the array is",d.size, d.shape)
    #d[d==0]= np.nan # convert zeros to nan
    return d
    

def read_radar_coords(filename):
    print("Reading textfile with CR positions...")
    site = []
    az = []
    rg = []
    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        print("File", filename, "exists and is readable.")
        f = open(filename)
        lines = f.readlines()
        idx = 0
        for line in lines:
            # get site name
            line = line.strip("\n")
            site.append(line.split("\t")[0])
            az.append(line.split("\t")[-2])
            rg.append(line.split("\t")[-1])
            idx = idx + 1  
        print("Radar coordinates at %d sites read" % (idx))
        f.close()
    else:
        print("ERROR: Can't read file", filename)
        sys.exit()
    print()
    return site, az, rg


def plot_results(sites, results, params):
    """
    Function to plot results of radar target response analysis, also outputs RCS values to file

    :param dict sites: containts site name and radar coordinates
    :param: dict results: containing paths to asc backscatter files (none if not given)
    :param dict params: config parameters
    """
    sitenames = sites.keys()
    for i in range(0, len(sitenames)):

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
            plot_mean_intensity2(avgI_a, avgI_d, cr_pos_a, cr_pos_d, name, params)

            # extract start and end date
            start_time = min(t_a[0], t_d[0])
            end_time = max(t_a[-1], t_d[-1])
            margin = (end_time - start_time) / 50
            start = start_time - margin
            end = end_time + margin

            # Plot average clutter time series
            plot_clutter2(t_a, t_d, clt_a, clt_d, start, end, name, params)
            # Plot scr time series
            plot_scr2(t_a, t_d, scr_a, scr_d, start, end, name, params)
            # Plot rcs time series
            plot_rcs2(t_a, t_d, rcs_a, rcs_d, start, end, name, params)

            # write RCS data to file
            filename_out = params[cf.OUT_DIR] + "/rcs_values_" + name + "_" + "Ascending.txt"
            fout = open(filename_out, 'w')
            for time, value in zip(t_a, rcs_a):
                timestr = time.strftime("%Y%m%d")
                fout.write("%s %f\n" % (timestr, value))
            fout.close()
            filename_out = params[cf.OUT_DIR] + "/rcs_values_" + name + "_" + "Descending.txt"
            fout = open(filename_out, 'w')
            for time, value in zip(t_d, rcs_d):
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
                geom = 'Ascending'
                cr = np.array([cr_new, [-999, -999]])
            if params[cf.DESC_LIST]:
                geom = 'Descending'
                cr = np.array([[-999, -999], cr_new])
            sites[name] = cr

            # Visualisation
            print('Site %s' % name)
            # Plot mean intensity image
            plot_mean_intensity(avgI, cr_pos, name, params)

            # extract start and end date
            start_time = t[0]
            end_time = t[-1]
            margin = (end_time - start_time) / 50
            start = start_time - margin
            end = end_time + margin

            # Plot average clutter time series
            plot_clutter(t, clt, start, end, name, geom, params)
            # Plot scr time series
            plot_scr(t, scr, start, end, name, geom, params)
            # Plot rcs time series
            plot_rcs(t, rcs, start, end, name, geom, params)

            # write RCS data to file
            filename_out = params[cf.OUT_DIR] + "/rcs_values_" + name + "_" + geom + ".txt"
            fout = open(filename_out, 'w')
            for time, value in zip(t, rcs):
                timestr = time.strftime("%Y%m%d")
                fout.write("%s %f\n" % (timestr, value))
            fout.close()


def write_radar_coords(filename_init, filename, sites, geom):
    print("Writing new CR positions to textfile...")
    fout = open(filename,'w')
    if os.path.isfile(filename_init) and os.access(filename_init, os.R_OK):
        print("File", filename_init, "exists and is readable.")
        f = open(filename_init)
        lines = f.readlines()
        for line in lines:
            # get site name
            name = line.split("\t")[0]
            cr = sites[name]
            temp = line.split("\t")[:-2]
            temp2 = "\t".join(temp)
            if geom == "asc":
                out_line = temp2 + "\t" + str(cr[0][1]) + \
                       "\t" + str(cr[0][0]) + "\n"
            if geom == "desc":
                out_line = temp2 + "\t" + str(cr[1][1]) + \
                       "\t" + str(cr[1][0]) + "\n"           
            fout.write(out_line)
        f.close()
        fout.close()    
    else:
        print("ERROR: Can't read file", filename_init)
        sys.exit()
    print()
    print("Azimuth and Range number of CR pixels has been written to " + \
    filename + ".")
    print()
    return


class Exception(Exception):
    """IO generic exception class"""