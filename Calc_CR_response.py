import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.colors import LogNorm
import glob, re
import numpy as np
from datetime import datetime
from coral.corner_reflector import *

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

    # set black/white colormap for plots
    cmap = plt.set_cmap('gist_gray')

    # draw new plot
    #fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig, ax1 = plt.subplots(1, 1, sharey=True)
    #ax = fig.add_subplot(1,1,1)
    cax = ax1.matshow(avgI_d, vmin=-20, vmax=10, cmap=cmap)
    #cax = ax2.matshow(avgI_d, vmin=-20, vmax=10, cmap=cmap)

    # define target window
    p1 = RegularPolygon(cr_pos, 4, (targ_win_sz/2)+1, \
                            orientation = np.pi / 4, linewidth=1, \
                            edgecolor='r',facecolor='none')
    # define clutter window
    p2 = RegularPolygon(cr_pos, 4, (clt_win_sz/2)+2, \
                            orientation = np.pi / 4, linewidth=1, \
                            edgecolor='y',facecolor='none')

    # add windows to plot
    ax1.add_patch(p1)
    ax1.add_patch(p2)

    # add text labels
    ax1.text(45, 42, name, color='w', fontsize=10)

    # plot labels
    ax1.set_xlabel('Range')
    ax1.set_ylabel('Azimuth')

    # add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('dB')

    # add title
    ax1.set_title('Mean intensity at %s' % name)

    # x-axis labels at bottom
    ax1.xaxis.set_tick_params(labeltop='off', labelbottom='on')

    # fit subplots and save fig
    fig.tight_layout()
    #fig.set_size_inches(w=6,h=4)

    # save PNG file
    fig.savefig('mean_intensity_%s.png' % name, dpi=300, bbox_inches='tight')

    # Plot RCS_SCR time series
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(t_d, rcs_d, 'ro-', label='RCS')
    plt.plot(t_d, scr_d, 'bo-', label='SCR')
    plt.xlim(start, end)
    plt.ylim(0, 40)
    plt.xlabel('Date')
    plt.ylabel('RCS / SCR (dB)')
    plt.legend(loc=4)
    plt.grid(True)
    plt.title('Corner Reflector response at %s' % name)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    fig.savefig('rcs_scr_%s.png' % name, dpi=300, bbox_inches='tight')


    # Plot average clutter time series
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(t_d, Avg_clt_d, 'bo-', label='Clutter')
    plt.xlim(start, end)
    plt.ylim(-16, -2)
    plt.xlabel('Date')
    plt.ylabel('Average Clutter (dB)')
    plt.legend(loc=1)
    plt.grid(True)
    plt.title('Average Clutter at %s' % name)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    fig.savefig('clutter_%s.png' % name, dpi=300, bbox_inches='tight')

