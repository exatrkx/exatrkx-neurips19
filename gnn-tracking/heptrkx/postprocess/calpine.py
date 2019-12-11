"""
Modules for plotting
"""
from .. import pairwise
import numpy as np
import matplotlib.pyplot as plt


fontsize = 16
minor_size=14
def plot_eff_vs_pt(sel_pt, true_pt, ax=None):
    if ax is None:
        fig, ax = plt.subplots()


    true_vals, bins, _ = ax.hist(true_pt, bins=np.linspace(0, 10, 21), histtype='step', lw=2, label='true')
    sel_vals,  bins, _ = ax.hist(sel_pt,  bins=np.linspace(0, 10, 21), histtype='step', lw=2,  label='selected')
    ax.set_xlabel('pT [GeV]', fontsize=fontsize)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax.set_ylabel('Counts', color='b', fontsize=fontsize)
    ax.tick_params('y', colors='b')
    ax.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax.legend(fontsize=fontsize)

    ax2 = ax.twinx()
    ratio = [x/y if y!=0 else 0 for x,y in zip(sel_vals, true_vals)]
    xvals = [0.5*(x[0]+x[1]) for x in pairwise(bins)]
    ax2.plot(xvals, ratio, '-o', lw=2, color='r')
    ax2.set_ylabel('Selected/True', color='r', fontsize=fontsize)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax2.tick_params('y',    colors='r')

    if ax is None:
        fig.tight_layout()
        return fig
    else:
        return None
