"""
This python module contains functions for calculating the response of
corner reflectors or other targets in Synthetic Aperture Radar images
"""
import re
import numpy as np
from datetime import datetime
from decimal import Decimal as D
import sys, os, os.path
from coral.dataio import readfile


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

        # read the SAR image and extract relevant metadata
        d[i], rho_r, rho_a, theta = readfile(g, sub_im, cr)

    # calculate mean Intensity image
    avgI = 10*np.log10(np.mean(d, axis=0))

    cr_pos = np.array([sub_im, sub_im])


    # find location of pixel with highest intensity inside target window
    # calculate target window bounds
    xmin_t = int(np.ceil(cr_pos[0] - targ_win_sz / 2))
    xmax_t = int(np.floor(cr_pos[0] + targ_win_sz / 2 + 1))
    ymin_t = int(np.ceil(cr_pos[1] - targ_win_sz / 2))
    ymax_t = int(np.floor(cr_pos[1] + targ_win_sz / 2 + 1))
    # crop mean intensity matrix to target window size
    avgI_t = avgI[ymin_t:ymax_t, xmin_t:xmax_t]
    # find matrix position of maximum mean intesity
    max_ix = np.unravel_index(np.argmax(avgI_t, axis=None), avgI_t.shape)
    # calculate shift w.r.t. central pixel
    centre_ix =  (targ_win_sz - 1) / 2
    shift = np.array([max_ix[1] - centre_ix, max_ix[0] - centre_ix], dtype=int)
    # updated cr position
    cr_new = cr + shift
    # also update cr_pos accordingly
    cr_pos = cr_pos + shift


    # calculate target energy
    En, Ncr = calc_integrated_energy(d, cr_pos, targ_win_sz)
    # calculate clutter energy for window centred in same spot as target window
    E1, N1 = calc_integrated_energy(d, cr_pos, clt_win_sz)
    # calculate average clutter
    Avg_clt, Eclt, Nclt = calc_clutter_intensity(En, E1, Ncr, N1)
    # Calculate total energy, signal-to-clutter ratio and radar cross section
    Ecr = calc_total_energy(Ncr, Nclt, Eclt, En)
    scr = calc_scr(Ecr, Eclt, Nclt)
    rcs = calc_rcs(Ecr, rho_r, rho_a, theta)

    return avgI, rcs, scr, Avg_clt, t, cr_new, cr_pos


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


