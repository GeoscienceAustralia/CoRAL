"""
This python module contains functions for reading Synthetic Aperture 
Radar images
"""
import numpy as np


def readfile(file, sub_im, cr):
    """Function to read an image file"""

    # read GAMMA 'par' file
    par = readpar(file + '.par')

    # extract relevant metadata
    rho_r = float(par['range_pixel_spacing'].split()[0])
    rho_a = float(par['azimuth_pixel_spacing'].split()[0])
    theta = float(par['incidence_angle'].split()[0])

    # open GAMMA flat-binary float file and read subset of image
    data = readmli(file, par, sub_im, cr)

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


