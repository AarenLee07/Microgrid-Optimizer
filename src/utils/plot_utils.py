import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm # colormap

import numpy as np
import pandas as pd

from datetime import datetime, timedelta


def customize_plt():
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12  # fontsize of legend
    plt.rcParams['legend.title_fontsize'] = 12  # fontsize of legend title
    plt.rcParams['legend.frameon'] = False # remove the box of legend
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.columnspacing'] = 1
    plt.rcParams['legend.labelspacing'] = 0.25
    plt.rcParams['axes.labelsize'] = 16  # fontsize of axes title
    plt.rcParams['axes.titlesize'] = 16  # fontsize of subplot title
    plt.rcParams['xtick.labelsize'] = 14  # fontsize of ticklabels
    plt.rcParams['ytick.labelsize'] = 14  # fontsize of ticklabels
    plt.rcParams['lines.linewidth'] = 2  # width of line
    # plt.rcParams['patch.linewidth'] = 2
customize_plt()

def rep_profile_visual(ys, prob, delta=0.25, cmap_name="RdYlGn", ref=None, max_ref=None, ax=None, skip=(0.4, 0.6)):
    """
    [Yi, 2023/03/03]
    
    How to inteprete:
    - line width: profile with high prob. is drawn with larger line width
    - color: if ref is None, color also indicates prob.
             if ref is not None, color indicates difference (L2-norm) with "central profile"
    Args:
        ys (np.array): shape = (S, K), S: # of clusters, K: # of steps per profile
        prob (np.array): shape = (S,)
        delta (float, optional): time interval, unit: hour. Defaults to 0.25.
        cmap_name (str, optional): colormap name. Defaults to "RdYlGn".
            suggest: "RdYlGn", "Spectral", "winter", "cool", etc. suffix "_r" for inverse.
            see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        ref (np.array, optional): central profile. used as reference to compute distance. Defaults to None.
        max_ref (float, optional): _description_. Defaults to None.
        ax (matplotlib.axes, optional): axes to draw. Defaults to None.
        skip (tuple, optional): interval of cmap to skip (color is too light). Default to (0.4, 0.6).
    """

    ax = ax if ax is not None else plt.gca()

    ys = ys[np.argsort(prob)]
    prob = np.sort(prob)
    

    S, K = ys.shape

    lw = np.clip(prob * S, 0.3, 5)
    zorder = prob * 100

    cmap = cm.get_cmap(cmap_name)

    if ref is None:
        color_dist = np.linspace(0,1,S)
    else:
        color_dist = np.sqrt(((ys - ref.reshape(1,-1)) ** 2).mean(axis=1))
        max_ref = max_ref if max_ref is not None else color_dist.max()
        color_dist = 1 - color_dist/max_ref

    # usually, in the middle of cmap is very light color (white , yellow, etc.)
    #   that we want to skip
    if skip is not None:
        lo, hi = skip
        mi = (lo+hi) / 2
        color_dist = (color_dist <= mi) * (color_dist *  (lo/mi)) +\
                    (color_dist > mi) * ((color_dist-mi) * ((1-hi)/(1-mi)) + hi)
         

    x = np.arange(0, K*delta, delta)

    for i in range(len(ys)):
        ax.plot(x, ys[i], lw=lw[i], color=cmap(color_dist[i]), zorder=zorder[i])

