"""
Ths is an example python script for running CoRAL analysis

Example usage: python3 Calc_CR_response.py
"""
import glob, re
import numpy as np
from datetime import datetime
from coral.corner_reflector import *
from coral.plot import *

# start/end date for plots
start = datetime(2018, 7, 1)
end = datetime(2018, 11, 1)

# Get list of image files in current directory
gtiffd = []
for file in glob.glob("data/*.mli"):
    gtiffd.append(file)

gtiffd.sort()

# final target positions manual entry
sites = {
#                       rg     az
    'SERF' : np.array([[ 87, 110]]),
}

#cropped image size
sub_im = 51

# define target_window size
targ_win_sz = 5

# define clutter window size
clt_win_sz = 9

#################################
def _inner(gtiff,sub_im,cr):
    # pre-allocate ndarray
    d = np.empty((len(gtiff),sub_im*2, sub_im*2))
    t = []

    for i, g in enumerate(gtiff):
        # get list of datetime objects
        m = re.search('data/(.+?)_VV', g)
        if m:
            t.append(datetime.strptime(m.group(1), "%Y%m%d"))

        print(i, g, g+'.par')
        # read par file
        par = readpar(g + '.par')
        #print(par)
        # open file and read subset of image
        d[i] = readmli(g, par, sub_im, cr)

    # calculate mean Intensity image
    avgI = 10*np.log10(np.mean(d, axis=0))

    #print("Incidence angle is:",par['incidence_angle'].split()[0])
    #print("range_pixel_spacing is:",par['range_pixel_spacing'].split()[0])
    #print("azimuth_pixel_spacing is:",par['azimuth_pixel_spacing'].split()[0])
    cr_pos = np.array([sub_im, sub_im])

    # calculate target energy, SCR and RCS
    En, Ncr, Eclt, Nclt, Avg_clt = calc_target_energy(d, cr_pos, targ_win_sz, clt_win_sz)
    Ecr = calc_total_energy(Ncr, Nclt, Eclt, En)
    scr = calc_scr(Ecr, Eclt, Nclt)
    rcs = calc_rcs(Ecr, par)

    return avgI, rcs, scr, Avg_clt, t

##################################
# loop through all sites
for name, cr in sites.items():
    print(name,'is desc',cr[0])

    avgI_d, rcs_d, scr_d, Avg_clt_d, t_d = _inner(gtiffd,sub_im,cr[0])

    cr_pos = np.array([sub_im, sub_im])

    # Plot mean intensity image
    plot_mean_intensity(avgI_d, cr_pos, targ_win_sz, clt_win_sz, name)

    # Plot RCS_SCR time series
    plot_rcs_scr(t_d, rcs_d, scr_d, start, end, name)

    # Plot average clutter time series
    plot_clutter(t_d, Avg_clt_d, start, end, name)

