"""
This python module contains functions for calculating the response of
corner reflectors or other targets in Synthetic Aperture Radar images
"""
import re
#from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from decimal import Decimal as D
import sys, os, os.path


def loop(files, sub_im, cr, targ_win_sz, clt_win_sz):
    """conduct CR processing for each site in a loop"""
    # pre-allocate ndarray
    d = np.empty((len(files),sub_im*2, sub_im*2))
    t = []

    for i, g in enumerate(files):
        # search for 8 character data string in file name
        m = re.search('\d{8}', g)
        if m:
            t.append(datetime.strptime(m.group(0), "%Y%m%d")) # convert to datetime object

        print(i, g, g+'.par')
        # read GAMMA 'par' file
        par = readpar(g + '.par')

        # open GAMMA float file and read subset of image
        d[i] = readmli(g, par, sub_im, cr)

    # calculate mean Intensity image
    avgI = 10*np.log10(np.mean(d, axis=0))

    #print("Incidence angle is:",par['incidence_angle'].split()[0])
    #print("range_pixel_spacing is:",par['range_pixel_spacing'].split()[0])
    #print("azimuth_pixel_spacing is:",par['azimuth_pixel_spacing'].split()[0])
    cr_pos = np.array([sub_im, sub_im])

    # calculate target energy
    En, Ncr = calc_integrated_energy(d, cr_pos, targ_win_sz)
    # calculate clutter energy for window centred in same spot as target window
    E1, N1 = calc_integrated_energy(d, cr_pos, clt_win_sz)
    # calculate average clutter
    Avg_clt, Eclt, Nclt = calc_clutter_intensity(En, E1, Ncr, N1)
    #
    Ecr = calc_total_energy(Ncr, Nclt, Eclt, En)
    scr = calc_scr(Ecr, Eclt, Nclt)

    rho_r = float(par['range_pixel_spacing'].split()[0])
    rho_a = float(par['azimuth_pixel_spacing'].split()[0])
    theta = float(par['incidence_angle'].split()[0])

    rcs = calc_rcs(Ecr, rho_r, rho_a, theta)

    return avgI, rcs, scr, Avg_clt, t


def get_win_bounds(pos, winsz):
    """Calculate sample window min/max bounds"""
    xmin = int(np.ceil(pos[0] - winsz / 2))
    xmax = int(np.floor(pos[0] + winsz / 2 + 1))
    ymin = int(np.ceil(pos[1] - winsz / 2))
    ymax = int(np.floor(pos[1] + winsz / 2 + 1))    

    return xmin, xmax, ymin, ymax


def calc_integrated_energy(d, pos, winsz):
    """Calculate the integrated energy within sample window"""
    # calculate window bounds
    xmin, xmax, ymin, ymax = get_win_bounds(pos, winsz)

    E = []
    N = []

    for i in range(d.shape[0]):
        # get image subset
        subd = d[i, ymin:ymax, xmin:xmax]

        E.append(subd.sum()) # total integrated energy in window
        N.append(subd.size) # number of samples in window

    #print("Total integrated energy in window is:",E)
    #print("Number of samples in window is:",N)
    return E, N


def calc_clutter_intensity(En, E, Ncr, N):
    """Calculate the average clutter intensity"""
    Avg_clt = []
    Eclt = []
    Nclt = []

    for i,item in enumerate(En):
        A = E[i] - En[i]
        B = N[i] - Ncr[i]
        Avg_clt.append(10*np.log10(A / B))
        Eclt.append(A)
        Nclt.append(B)

    #print("Average clutter intensity is:",Avg_clt, "decibels")
    return Avg_clt, Eclt, Nclt


def calc_total_energy(Ncr, Nclt, Eclt, En):
    """Calculate the total integrated energy in the target impulse response"""
    Ecr = []

    for i in range(len(Ncr)):
        # Garthwaite 2017 Equation 7
        Ecr.append(En[i] - (float(D(Ncr[i])/D(Nclt[i])) * Eclt[i]))

    #print("Total integrated target energy is:",Ecr)
    return Ecr


def calc_scr(Ecr, Eclt, Nclt):
    """Calculate the Signal to Clutter Ratio in decibels"""
    scr_db = []

    for i in range(len(Ecr)):
        # Signal to Clutter Ratio (Garthwaite 2017 Equation 7)
        scr = Ecr[i] / (Eclt[i] / Nclt[i])
        #print("SCR is ",scr, Ecr[i], Eclt[i], Nclt[i])
        # Re-assign negative SCR to zero dB
        if scr < 1: scr = 1
        scr_db.append(10 * np.log10(scr))

    #print("Target SCR is:",scr_db,"dB",scr)
    return scr_db


def calc_rcs(Ecr, rho_r, rho_a, theta):
    """Calculate the Radar Cross Section of the target in decibels"""
    # illuminated area (Garthwaite 2017 Equation 2)
    A = (rho_r * rho_a) / np.sin((theta/180) * np.pi)

    #print("The illuminated pixel area is:",A,"m^2")
    rcs_db = []

    for i in range(len(Ecr)):
        # Radar Cross Section (Garthwaite 2017 Equation 8)
        rcs = Ecr[i] * A
        #print("RCS is ",rcs, Ecr[i])
        # Re-assign negative RCS to zero dB
        if rcs < 1: rcs = 1
        rcs_db.append(10 * np.log10(rcs))

    #print("target RCS is:",rcs_db,"dBsm")
    return rcs_db


################################
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

################################
def readmli(datafile, par, sub_im, cr):
    """Function to read a GAMMA mli file and provide a subsetted image"""
    ct = int(par['range_samples']) * int(par['azimuth_lines'])

    dt = np.dtype('>f4') # GAMMA files are big endian 32 bit float

    d = np.fromfile(datafile, dtype=dt, count=ct)

    d = d.reshape(int(par['azimuth_lines']), int(par['range_samples']))
    #print("Number of elements and size of the array is",d.size, d.shape)
    #d[d==0]= np.nan # convert zeros to nan
    return d[cr[1]-sub_im:cr[1]+sub_im,cr[0]-sub_im:cr[0]+sub_im]


