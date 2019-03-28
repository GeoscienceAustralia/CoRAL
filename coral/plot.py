"""
This python module contains functions for plotting CoRAL output
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.colors import LogNorm
import numpy as np


def plot_mean_intensity(avgI, cr_pos, targ_win_sz, clt_win_sz, name):
    '''Plot image of mean SAR intensity'''
    # set black/white colormap for plots
    cmap = plt.set_cmap('gist_gray')

    # draw new plot
    #fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig, ax1 = plt.subplots(1, 1, sharey=True)
    #ax = fig.add_subplot(1,1,1)
    cax = ax1.matshow(avgI, vmin=-20, vmax=10, cmap=cmap)
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
    #ax1.text(45, 42, name, color='w', fontsize=10)

    # plot labels
    ax1.set_xlabel('Range')
    ax1.set_ylabel('Azimuth')

    # add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('dB')

    # add title
    ax1.set_title('Mean intensity at %s' % name)

    # x-axis labels at bottom
    ax1.xaxis.set_tick_params(labeltop='False', labelbottom='True')

    # fit subplots and save fig
    fig.tight_layout()
    #fig.set_size_inches(w=6,h=4)

    # save PNG file
    fig.savefig('mean_intensity_%s.png' % name, dpi=300, bbox_inches='tight')

    return


def plot_rcs_scr(t, rcs, scr, start, end, name):
    '''Plot RCS and SCR time series'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(t, rcs, 'ro-', label='RCS')
    plt.plot(t, scr, 'bo-', label='SCR')
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

    return

def plot_clutter(t, clt, start, end, name):
    '''Plot average clutter time series'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(t, clt, 'bo-', label='Clutter')
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

    return
