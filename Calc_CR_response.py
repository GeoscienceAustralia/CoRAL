"""
Ths is an example python script for running CoRAL analysis

Example command line usage: python3 Calc_CR_response.py
"""
import glob
import numpy as np
from datetime import datetime
from coral.corner_reflector import loop
from coral.plot import *

# start/end date for plots
start = datetime(2018, 7, 1)
end = datetime(2018, 11, 1)

# Get list of image files in current directory
files = []
for file in glob.glob("data/*.mli"):
    files.append(file)

files.sort()

# final target positions manual entry
sites = {
#   site name            rg   az
    'SERF' : np.array([[ 87, 110]]),
}

#cropped image size
sub_im = 51

# define target_window size
targ_win_sz = 5

# define clutter window size
clt_win_sz = 9

# loop through all corner reflector sites
for name, cr in sites.items():
    print(name,'is desc',cr[0])

    avgI_d, rcs_d, scr_d, Avg_clt_d, t_d = loop(files, sub_im, cr[0], targ_win_sz, clt_win_sz)

    cr_pos = np.array([sub_im, sub_im])

    # Plot mean intensity image
    plot_mean_intensity(avgI_d, cr_pos, targ_win_sz, clt_win_sz, name)

    # Plot RCS_SCR time series
    plot_rcs_scr(t_d, rcs_d, scr_d, start, end, name)

    # Plot average clutter time series
    plot_clutter(t_d, Avg_clt_d, start, end, name)

