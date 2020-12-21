"""
This python module contains functions for plotting CoRAL output
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
from coral import config as cf


def plot_mean_intensity2(avgI_a, avgI_d, cr_pos_a, cr_pos_d, name, params):
    '''Plot image of mean SAR intensity'''
    # set black/white colormap for plots
    cmap = plt.set_cmap('gist_gray')

    # draw new plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,5))
    axlist = [ax1, ax2]
    im1 = ax1.matshow(avgI_d, vmin=-20, vmax=10, cmap=cmap)
    im2 = ax2.matshow(avgI_a, vmin=-20, vmax=10, cmap=cmap)

    # define target window
    p1 = RegularPolygon(cr_pos_d, 4, (params[cf.TARG_WIN_SZ]/2)+2, \
                        orientation = np.pi / 4, linewidth=1, \
                        edgecolor='r',facecolor='none')
    # define clutter window
    p2 = RegularPolygon(cr_pos_d, 4, (params[cf.CLT_WIN_SZ]/2)+3, \
                        orientation = np.pi / 4, linewidth=1, \
                        edgecolor='y',facecolor='none')
    # define target window
    p3 = RegularPolygon(cr_pos_a, 4, (params[cf.TARG_WIN_SZ]/2)+2, \
                        orientation = np.pi / 4, linewidth=1, \
                        edgecolor='r',facecolor='none')
    # define clutter window
    p4 = RegularPolygon(cr_pos_a, 4, (params[cf.CLT_WIN_SZ]/2)+3, \
                        orientation = np.pi / 4, linewidth=1, \
                        edgecolor='y',facecolor='none')
    # add windows to plot
    ax1.add_patch(p1)
    ax1.add_patch(p2)
    ax2.add_patch(p3)
    ax2.add_patch(p4)

    # add text labels
    ax1.text(45, 42, name, color='w', fontsize=10)
    ax2.text(45, 42, name, color='w', fontsize=10)
    # plot labels
    ax1.set_xlabel('Range')
    ax2.set_xlabel('Range')
    ax1.set_ylabel('Azimuth')
    # add colorbar
    cbar = fig.colorbar(im1, ax=axlist)
    cbar.set_label('dB')
    # add title
    #fig.set_title('Mean intensity at site %s' % name)
    ax1.set_title('Descending')
    ax2.set_title('Ascending')
    # x-axis labels at bottom
    ax1.xaxis.set_tick_params(labeltop='off', labelbottom='on')
    ax2.xaxis.set_tick_params(labeltop='off', labelbottom='on')

    # save PNG file
    filename = params[cf.OUT_DIR] + "/mean_intensity_" + name + ".png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    # avoid "RuntimeWarning: More than 20 figures have been opened"
    # by closing all open figures
    plt.close('all')
    
    return


def plot_clutter2(t_a, t_d, clt_a, clt_d, start, end, name, params):
    '''Plot average clutter time series'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(t_a, clt_a, 'ro-', label='Asc')
    plt.plot(t_d, clt_d, 'bo-', label='Desc')
    plt.xlim(start, end)
    plt.ylim(params[cf.YMIN_CLUTTER], params[cf.YMAX_CLUTTER])
    plt.xlabel('Date')
    plt.ylabel('Average Clutter (dB)')
    plt.legend(loc=1)
    plt.grid(True)
    plt.title('Average Clutter at site %s' % name)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        
    # save PNG file   
    filename = path_out + "/clutter_" + name + ".png"    
    fig.savefig(filename, dpi=300, bbox_inches='tight')

    # avoid "RuntimeWarning: More than 20 figures have been opened"
    # by closing all open figures
    plt.close('all')

    return
    
    
def plot_scr2(t_a, t_d, scr_a, scr_d, start, end, name, params):
    '''Plot RCS time series'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(t_a, scr_a, 'ro-', label='Ascending')
    plt.plot(t_d, scr_d, 'bo-', label='Descending')
    plt.xlim(start, end)
    plt.ylim(0, params[cf.YMAX_SCR])
    plt.xlabel('Date')
    plt.ylabel('SCR (dB)')
    plt.legend(loc=4)
    plt.grid(True)
    plt.title('Signal to Clutter Ratio at site %s' % name)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        
    # save PNG file   
    filename = path_out + "/scr_" + name + ".png"       
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    # avoid "RuntimeWarning: More than 20 figures have been opened"
    # by closing all open figures
    plt.close('all')
    
    return    
    
    
def plot_rcs2(t_a, t_d, rcs_a, rcs_d, start, end, name, params):
    '''Plot RCS time series'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(t_a, rcs_a, 'ro-', label='Ascending')
    plt.plot(t_d, rcs_d, 'bo-', label='Descending')
    plt.xlim(start, end)
    plt.ylim(0, params[cf.YMAX_RCS])
    plt.xlabel('Date')
    plt.ylabel('RCS (dB$\mathregular{m^2}$)')
    plt.legend(loc=4)
    plt.grid(True)
    plt.title('Radar Cross Section at site %s' % name)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        
    # save PNG file   
    filename = path_out + "/rcs_" + name + ".png"       
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    # avoid "RuntimeWarning: More than 20 figures have been opened"
    # by closing all open figures
    plt.close('all')
    
    return
    
