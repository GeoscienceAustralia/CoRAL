from __future__ import print_function
import glob, re
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from decimal import Decimal as D
import sys, os, os.path



def calc_target_energy(d, cr_pos, targ_win_sz, clt_win_sz):

    # calculate target window bounds
    xmin_t = int(np.ceil(cr_pos[0] - targ_win_sz / 2))
    xmax_t = int(np.floor(cr_pos[0] + targ_win_sz / 2 + 1))
    ymin_t = int(np.ceil(cr_pos[1] - targ_win_sz / 2))
    ymax_t = int(np.floor(cr_pos[1] + targ_win_sz / 2 + 1))

    # calculate clutter window bounds
    xmin_c = int(np.ceil(cr_pos[0] - clt_win_sz / 2))
    xmax_c = int(np.floor(cr_pos[0] + clt_win_sz / 2 + 1))
    ymin_c = int(np.ceil(cr_pos[1] - clt_win_sz / 2))
    ymax_c = int(np.floor(cr_pos[1] + clt_win_sz / 2 + 1))

    En = []
    Ncr = []
    Eclt = []
    Nclt = []
    Avg_clt = []

    for i in range(d.shape[0]):
        # target subset
        subd_t = d[i, ymin_t:ymax_t, xmin_t:xmax_t]

        #plt.matshow(subd_t)
        #plt.colorbar()
        #plt.show()

        # Garthwaite 2017 Equation 7
        En.append(subd_t.sum()) # total integrated target energy
        Ncr.append(subd_t.size) # number of samples in target window

        # clutter subset
        subd_c = d[i, ymin_c:ymax_c, xmin_c:xmax_c]

        A = subd_c.sum() - subd_t.sum()
        Eclt.append(A)
        B = subd_c.size - subd_t.size
        Nclt.append(B)
        Avg_clt.append(10*np.log10(A / B))

    print("Total integrated energy in target window is:",En)
    #print("Number of samples in target window is:",Ncr)
    #print("Average clutter intensity is:",Avg_clt, "decibels")

    return En, Ncr, Eclt, Nclt, Avg_clt


def calc_clutter(d, clt_pos, clt_win_sz):
    # calculate second image bounds
    xmin = int(np.ceil(clt_pos[0] - clt_win_sz / 2))
    xmax = int(np.floor(clt_pos[0] + clt_win_sz / 2 + 1))
    ymin = int(np.ceil(clt_pos[1] - clt_win_sz / 2))
    ymax = int(np.floor(clt_pos[1] + clt_win_sz / 2 + 1))

    print(xmin, xmax, ymin, ymax)
    Eclt = []
    Nclt = []
    Avg_clt = []

    for i in range(d.shape[0]):
        # subset of full data array around defined CR
        subd = d[i, ymin:ymax, xmin:xmax]

        #plt.matshow(subd)
        #plt.colorbar()
        #plt.show()

        Eclt.append(subd.sum())
        Nclt.append(subd.size)
        Avg_clt.append(10*np.log10(subd.sum() / subd.size))

    #print("Average clutter intensity is:",Avg_clt, "decibels")
    return Eclt, Nclt, Avg_clt


def calc_total_energy(Ncr, Nclt, Eclt, En):
    Ecr = []

    for i in range(len(Ncr)):
        # Garthwaite 2017 Equation 7
        Ecr.append(En[i] - (float(D(Ncr[i])/D(Nclt[i])) * Eclt[i]))

    #print("Total integrated target energy is:",Ecr)
    return Ecr


def calc_scr(Ecr, Eclt, Nclt):
    scr_db = []

    for i in range(len(Ecr)):
        # Signal to Clutter Ratio (Garthwaite 2017 Equation 7)
        scr = Ecr[i] / (Eclt[i] / Nclt[i])
        print("SCR is ",scr, Ecr[i], Eclt[i], Nclt[i])
        scr_db.append(10 * np.log10(scr))

    print("Target SCR is:",scr_db,"dB",scr)
    return scr_db


def calc_rcs(Ecr, par):
    rho_r = float(par['range_pixel_spacing'].split()[0])
    rho_a = float(par['azimuth_pixel_spacing'].split()[0])
    theta = float(par['incidence_angle'].split()[0])

    # illuminated area (Garthwaite 2017 Equation 2)
    A = (rho_r * rho_a) / np.sin((theta/180) * np.pi)

    #print("The illuminated pixel area is:",A,"m^2")
    rcs_db = []

    for i in range(len(Ecr)):
        # Radar Cross Section (Garthwaite 2017 Equation 8)
        rcs = Ecr[i] * A
        print("RCS is ",rcs, Ecr[i])
        rcs_db.append(10 * np.log10(rcs))

    #print("target RCS is:",rcs_db,"dBsm")
    return rcs_db


################################
def readpar(file):
    """Function to read a GAMMA par file into a dictionary"""
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
    print("Number of elements and size of the array is",d.size, d.shape)
    #d[d==0]= np.nan # convert zeros to nan
    return d[cr[1]-sub_im:cr[1]+sub_im,cr[0]-sub_im:cr[0]+sub_im]


################################
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
            site.append(line.split("\t")[0])
            az.append(line.split("\t")[5])
            rg.append(line.split("\t")[6])
            idx = idx + 1
        print("Radar coordinates at %d sites read" % (idx))
    else:
        print("ERROR: Can't read file", filename)
        sys.exit()
    print()
    return site, az, rg

