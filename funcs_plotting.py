import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

def plot_W(W_pred_arr, W):
    ## Crossplot energy (density)
    fig = plt.figure(figsize=(3, 3))
    plt.gcf().patch.set_facecolor("None")
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.scatter(W, W_pred_arr, s=1, alpha=0.3)
    plt.grid()
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.gca().axline((0,0), (1,1), c='black', zorder=-1)
    plt.title('$\mathfrak{W}$')
    plt.xlabel('real')
    plt.ylabel('predicted')
    ax = plt.gca()

    x_lims = ax.get_xlim()
    x_diff = x_lims[1] - x_lims[0]
    y_lims = ax.get_ylim()
    y_diff = y_lims[1] - y_lims[0]
    if 0.2 < x_diff/y_diff < 5.0:
        ax.set_aspect('equal')
        ax.apply_aspect()

        # get x-tick interval
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xstep = np.diff(xticks[[0,1]])[0]
        ystep = np.diff(yticks[[0,1]])[0]

        # if the difference between the step sizes is not too large
        if 0.2 < xstep/ystep < 5.0:
            # Change major ticks to be the same for x and y
            ax.yaxis.set_major_locator(MultipleLocator(xstep))

    return fig

def plot_P(P_pred_arr, P):
    # Crossplots P for each component
    fig, axes = plt.subplots(2,2,figsize=(5, 5))
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    fig.patch.set_facecolor("None")
    titles = ['xx', 'xy', 'yx', 'yy']
    for i, axes2 in enumerate(axes):
        for j, ax in enumerate(axes2):
            comp = '$P_{'+titles[i*2+j]+'}$'
            ax.scatter(P[:, i, j], P_pred_arr[:, i, j], s=1, alpha=0.3)
            ax.grid()
            # plt.yscale('log')
            # plt.xscale('log')
            # ax.axline((0,0), (1,1), c='black', zorder=-1)
            ax.set_title(comp)
            ax.set_xlabel('real')
            ax.set_ylabel('predicted')

            x_lims = ax.get_xlim()
            x_diff = x_lims[1] - x_lims[0]
            y_lims = ax.get_ylim()
            y_diff = y_lims[1] - y_lims[0]
            if 0.2 < x_diff/y_diff < 5.0:
                ax.set_aspect('equal')
                ax.apply_aspect()

                # get x-tick interval
                xticks = ax.get_xticks()
                yticks = ax.get_yticks()
                xstep = np.diff(xticks[[0,1]])[0]
                ystep = np.diff(yticks[[0,1]])[0]

                # if the difference between the step sizes is not too large
                if 0.2 < xstep/ystep < 5.0:
                    # Change major ticks to be the same for x and y
                    ax.yaxis.set_major_locator(MultipleLocator(xstep))
    return fig

def plot_D(D_pred_arr, D):
    # Crossplots D for each component
    D_temp = D.reshape(-1, 4, 4)
    D_pred_temp = D_pred_arr.reshape(-1, 4, 4)
    fig, axes = plt.subplots(4, 4,figsize=(9,9))
    fig.patch.set_facecolor("None")
    plt.subplots_adjust(hspace=0.7, wspace=0.7)
    titles = ['xx', 'xy', 'yx', 'yy']
    for i in range(4):
        for j in range(4):
            if i > j:
                ax = axes[i,j]
                ax.axis('off')
            else:
                ax = axes[i, j]
                comp = '$D_{' + titles[i] + titles[j] + '}$'
                ax.scatter(D_temp[:, i, j], D_pred_temp[:, i, j], s=1, alpha=0.3)
                ax.grid('both')
                # plt.yscale('log')
                # plt.xscale('log')
                # ax.axline((0,0), (1,1), c='black', zorder=-1)
                ax.set_title(comp)
                ax.set_xlabel('real')
                ax.set_ylabel('predicted')

                x_lims = ax.get_xlim()
                x_diff = x_lims[1] - x_lims[0]
                y_lims = ax.get_ylim()
                y_diff = y_lims[1] - y_lims[0]
                if 0.2 < x_diff/y_diff < 5.0:
                    ax.set_aspect('equal')
                    ax.apply_aspect()

                    # get x-tick interval
                    xticks = axes[i,j].get_xticks()
                    yticks = axes[i,j].get_yticks()
                    xstep = np.diff(xticks[[0,1]])[0]
                    ystep = np.diff(yticks[[0,1]])[0]

                    # if the difference between the step sizes is not too large
                    if 0.2 < xstep/ystep < 5.0:
                        # Change major ticks to be the same for x and y
                        ax.yaxis.set_major_locator(MultipleLocator(xstep))

    return fig