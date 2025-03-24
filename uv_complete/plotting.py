import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_array(antarray, fig=None, ax=None):
    """Plots dynamic state of the array and uv coverage after each antenna addition."""
    

    # Unpack everything from the AntArray
    antpos = antarray.antpos
    commanded = antarray.commanded
    diameter = antarray.diameter
    
    history = antarray.history
    n_new_fulfilled_list = history["n_new_fulfilled"]
    n_not_fulfilled_list = history["n_not_fulfilled"]
    new_fulfilled_list = history["new_fulfilled"]
    step_time_array = history["step_time"]
    efficiency_array = history["efficiency"]

    
    #if fig is None or ax is None:
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    colormap = cm.viridis

    # === Top-left: uv plane ===
    ax[0, 0].cla()
    ax[0, 0].set_title('uv plane')
    ax[0, 0].set_xlabel(r'$u$ [m]')
    ax[0, 0].set_ylabel(r'$v$ [m]')
    ax[0, 0].plot(commanded[:, 0], commanded[:, 1], '.', markersize=2, color='k', alpha=1, zorder=0)

    for i, new_fulfilled in enumerate(new_fulfilled_list):
        if new_fulfilled is not None and len(new_fulfilled.shape) == 2:
            ax[0, 0].plot(new_fulfilled[:, 0], new_fulfilled[:, 1], '.', markersize=2, color=colormap(i / len(new_fulfilled_list)), zorder=1)

    ax[0, 0].set_aspect('equal')

    # === Bottom-left: antenna layout ===
    ax[1, 0].cla()
    ax[1, 0].set_title(f'Array ({len(antpos)} antennas)')
    ax[1, 0].set_xlabel('EW [m]')
    ax[1, 0].set_ylabel('NS [m]')
    ax[1, 0].set_aspect('equal')
    color_scale = np.linspace(0, 1, len(antpos))
    marker_size = ((diameter / 2) * 72 / fig.dpi) ** 2 if diameter else 10
    scatter = ax[1, 0].scatter(antpos[:, 0], antpos[:, 1], c=colormap(color_scale), s=marker_size)
    fig.colorbar(scatter, ax=ax[0, 0], orientation='horizontal', pad=0.2).set_label('Antenna rank')

    # === Top-right: newly fulfilled vs remaining ===
    ax[0, 1].cla()
    ax[0, 1].plot(np.arange(1, len(n_new_fulfilled_list) + 1), n_new_fulfilled_list, color='b')
    ax[0, 1].set_ylabel('Newly fulfilled', color='b')
    ax[0, 1].set_xlabel('Antenna rank')
    ax[0, 1].grid()
    ax_remaining = ax[0, 1].twinx()
    ax_remaining.plot(np.arange(1, len(n_not_fulfilled_list) + 1), n_not_fulfilled_list, color='r')
    ax_remaining.set_ylabel('UV points remaining', color='r')

    # === Bottom-right: efficiency and step time ===
    ax[1, 1].cla()
    ax[1, 1].set_xlabel('Antenna rank')
    ax[1, 1].grid()

    if step_time_array is not None:
        ax[1, 1].plot(np.arange(1, len(step_time_array) + 1), step_time_array, color='b')
        ax[1, 1].set_ylabel('Time [s]', color='b')

    if efficiency_array is not None:
        ax_eff = ax[1, 1].twinx()
        ax_eff.plot(np.arange(1, len(efficiency_array) + 1), efficiency_array, color='r')
        ax_eff.set_ylabel('Efficiency', color='r')

    plt.tight_layout()
    plt.pause(0.01)
    plt.show()
    return fig, ax
