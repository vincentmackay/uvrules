import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from finufft import nufft2d1
from .geometry import antpos_to_uv


def plot_array(AA,
               show_commanded: bool = False,
               figsize=(10, 5),
               dpi=150,
               baseline_units: str = 'm',
               fewer_points: bool = True,
               color_antpos='tab:blue',
               color_uv='black',
               color_commanded='red'):
    """
    Plot antenna layout and baseline uv coverage.

    Parameters
    ----------
    AA : AntArray
        AntArray instance with antpos and (optionally) commanded.

    show_commanded : bool, optional
        Whether to overlay commanded uv points.

    figsize : tuple, optional
        Figure size in inches.

    dpi : int, optional
        Figure resolution in dots per inch.

    baseline_units : str, optional
        Either 'm' (meters) or 'wl' (wavelengths). Controls uv scaling.

    fewer_points : bool, optional
        If True, downsample uv points for visibility.

    color_antpos : str
        Antenna dot color.

    color_uv : str
        Baseline dot color.

    color_commanded : str
        Commanded uv point color.
    """
    if baseline_units not in ['m', 'wl']:
        raise ValueError("baseline_units must be 'm' or 'wl'.")

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    uvs = antpos_to_uv(AA.antpos)
    if baseline_units == 'wl':
        uvs /= AA.ref_wl

    n_skip = int(10 * np.floor(np.log10(len(uvs)))) if fewer_points else 1
    axes[0].scatter(uvs[::n_skip, 0], uvs[::n_skip, 1], s=1, alpha=0.8, color=color_uv, label='Baselines')

    if show_commanded and hasattr(AA, "commanded") and AA.commanded is not None:
        axes[0].scatter(AA.commanded[:, 0], AA.commanded[:, 1], s=1, alpha=0.5, color=color_commanded, label='Commanded')

    axes[0].set_title(r'$uv$ coverage' + (f' (1/{n_skip} points)' if fewer_points else ''))
    unit_label = rf"$\lambda$ @ {AA.ref_freq * 1e-9:.2g} GHz" if baseline_units == 'wl' else 'm'
    axes[0].set_xlabel(rf"$u$ [{unit_label}]")
    axes[0].set_ylabel(rf"$v$ [{unit_label}]")
    axes[0].set_aspect('equal')
    axes[0].legend()

    axes[1].scatter(AA.antpos[:, 0], AA.antpos[:, 1], s=1, color=color_antpos)
    axes[1].set_title('Antenna layout')
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('y [m]')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    return fig, axes


def plot_history(AA, fig=None, ax=None):
    """
    Plot dynamic evolution of uv fulfillment, layout, and efficiency.

    Parameters
    ----------
    AA : AntArray
        Array instance with .history attributes populated.

    fig : matplotlib.figure.Figure or None
        Optional existing figure to reuse.

    ax : np.ndarray of Axes or None
        Optional 2x2 array of subplots.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    antpos = AA.antpos
    commanded = AA.commanded
    diameter = AA.diameter
    hist = AA.history

    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    colormap = cm.viridis

    # Top-left: uv fulfillment by step
    ax[0, 0].cla()
    ax[0, 0].set_title('uv plane')
    ax[0, 0].set_xlabel(r'$u$ [m]')
    ax[0, 0].set_ylabel(r'$v$ [m]')
    ax[0, 0].plot(commanded[:, 0], commanded[:, 1], '.', markersize=2, color='k', alpha=1, zorder=0)
    for i, uv in enumerate(hist["new_fulfilled"]):
        if uv is not None and uv.ndim == 2:
            ax[0, 0].plot(uv[:, 0], uv[:, 1], '.', markersize=2, color=colormap(i / len(hist["new_fulfilled"])), zorder=1)
    ax[0, 0].set_aspect('equal')

    # Bottom-left: antenna layout
    ax[1, 0].cla()
    ax[1, 0].set_title(f'Array ({len(antpos)} antennas)')
    ax[1, 0].set_xlabel('EW [m]')
    ax[1, 0].set_ylabel('NS [m]')
    ax[1, 0].set_aspect('equal')
    marker_size = ((diameter / 2) * 72 / fig.dpi) ** 2 if diameter else 10
    cmap_vals = np.linspace(0, 1, len(antpos))
    scatter = ax[1, 0].scatter(antpos[:, 0], antpos[:, 1], c=colormap(cmap_vals), s=marker_size)
    fig.colorbar(scatter, ax=ax[0, 0], orientation='horizontal', pad=0.2).set_label('Antenna rank')

    # Top-right: newly fulfilled and remaining
    ax[0, 1].cla()
    ax[0, 1].plot(np.arange(1, len(hist["n_new_fulfilled"]) + 1), hist["n_new_fulfilled"], color='b')
    ax[0, 1].set_ylabel(r'Number of $uv$ points fulfilled at last step', color='b')
    ax[0, 1].set_xlabel('Antenna index')
    ax[0, 1].grid(True)
    ax2 = ax[0, 1].twinx()
    ax2.plot(np.arange(1, len(hist["n_not_fulfilled"]) + 1), hist["n_not_fulfilled"], color='r')
    ax2.set_ylabel('Remaining uv points', color='r')

    # Bottom-right: step time and efficiency
    ax[1, 1].cla()
    ax[1, 1].set_xlabel('Antenna index')
    ax[1, 1].grid(True)
    if hist["step_time"] is not None:
        ax[1, 1].plot(np.arange(1, len(hist["step_time"]) + 1), hist["step_time"], color='b')
        ax[1, 1].set_ylabel('Time per step [s]', color='b')
    if hist["efficiency"] is not None:
        ax_eff = ax[1, 1].twinx()
        ax_eff.plot(np.arange(1, len(hist["efficiency"]) + 1), hist["efficiency"], color='r')
        ax_eff.set_ylabel('Efficiency', color='r')

    plt.tight_layout()
    return fig, ax


def compute_beam_nufft_old(uvs: np.ndarray, n_px_side=100, fov_range=90, vmin=-8, dimensions=1):
    """
    Compute synthesized beam from uv points using NUFFT.

    Parameters
    ----------
    uvs : ndarray
        UV coordinates (wavelengths), shape (n, 2).
    n_px_side : int
        Image resolution (pixels per side).
    fov_range : float
        Half-field of view in degrees.
    vmin : float
        Dynamic range floor.
    dimensions : int
        1 or 2 depending on output beam shape.

    Returns
    -------
    psf : 2D array
        Power beam (normalized).
    sin_theta : 1D array
        Angular grid in sin(theta).
    """
    uvs_full = np.vstack([uvs, -uvs])
    max_uv = np.max(np.linalg.norm(uvs_full, axis=1))
    resolution_factor = (n_px_side - 1) / (4 * max_uv * np.sin(np.radians(fov_range)))

    u_scaled = np.pi * uvs_full[:, 0] / resolution_factor
    v_scaled = np.pi * uvs_full[:, 1] / resolution_factor
    vis = np.ones(len(u_scaled), dtype=np.complex128)
    eps = max(1e-16, 10 ** (vmin - 2))
    image = nufft2d1(u_scaled, v_scaled, vis, (n_px_side, n_px_side), eps=eps, isign=1)

    psf = np.abs(image.reshape((n_px_side, n_px_side)))
    psf /= psf.max()
    delta_sin_theta = 1 / (2 * max_uv)
    sin_theta = (np.arange(n_px_side) - n_px_side // 2) * delta_sin_theta / resolution_factor
    return psf, sin_theta
def plot_psf_1d_old(
    AA=None,
    uvs=None,
    n_px_side=1000,
    angle_range_deg=90,
    angle_range_arcmin=None,
    vmin=-8,
    vmax=0,
    az_avg=True,
    fig=None,
    ax=None,
    custom_label='',
    color='k',
    scale='power',
    new_fig=False,
    nbins=128,
    show_fwhm=True,
    show_legend=True,
    **plot_kwargs,
):
    """
    Plot 1D cut or azimuthal average of synthesized beam.

    Parameters
    ----------
    AA : AntArray, optional
        AntArray with antpos and ref_wl.
    uvs : ndarray, optional
        UV coordinates, shape (n, 2).
    n_px_side : int
        Resolution of beam image.
    angle_range_deg : float
        Field of view in degrees.
    angle_range_arcmin : float, optional
        Overrides degrees if given.
    vmin, vmax : float
        Dynamic range for log scale.
    az_avg : bool
        Whether to plot azimuthal average instead of 1D cut.
    fig, ax : Figure and Axis
        Optional reuse of axis.
    custom_label : str
        Custom plot label.
    color : str
        Plot color.
    scale : {'power', 'linear'}
        Output scaling.
    new_fig : bool
        Force new figure.
    nbins : int
        Number of radial bins.
    show_fwhm : bool
        Display estimated FWHM.
    show_legend : bool
        Display legend.
    plot_kwargs : dict
        Extra arguments to pass to plot.

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    if AA is not None:
        uvs = antpos_to_uv(AA.antpos) / AA.ref_wl

    fov = angle_range_arcmin / 60 if angle_range_arcmin else angle_range_deg
    psf, sin_theta = compute_beam_nufft(uvs, n_px_side=n_px_side, fov_range=fov)
    if scale == 'power':
        psf **= 2

    theta = np.arcsin(np.clip(sin_theta, -1, 1)) * (180 / np.pi)
    if angle_range_arcmin:
        theta *= 60
        theta_label = r"$\theta$ [arcmin]"
    else:
        theta_label = r"$\theta$ [deg]"

    cut = psf[:, psf.shape[1] // 2]
    THETA_X, THETA_Y = np.meshgrid(theta, theta)
    R = np.sqrt(THETA_X ** 2 + THETA_Y ** 2).flatten()
    B = psf.flatten()
    valid = np.isfinite(R) & np.isfinite(B)
    R = R[valid]
    B = B[valid]

    max_r = np.max(R)
    bin_width = max_r / (nbins // 2)
    bin_centers = np.arange(-nbins // 2, nbins // 2 + 1) * bin_width
    bins = np.append(bin_centers - bin_width / 2, bin_centers[-1] + bin_width / 2)
    i_zero = np.argmin(np.abs(bin_centers))
    digitized = np.clip(np.digitize(R, bins) - 1, 0, len(bin_centers) - 1)

    avg = np.bincount(digitized, weights=B, minlength=len(bin_centers))
    counts = np.bincount(digitized, minlength=len(bin_centers))
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = avg / counts
        avg[counts == 0] = np.nan
    avg[i_zero] = 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        new_fig = True

    if az_avg:
        label = custom_label or "Az avg"
        if show_fwhm:
            half_max = 0.5
            idx = np.where(avg < half_max)[0]
            if len(idx) > 0 and idx[0] > 0:
                i = idx[0]
                x1, x2 = bin_centers[i - 1], bin_centers[i]
                y1, y2 = avg[i - 1], avg[i]
                slope = (y2 - y1) / (x2 - x1)
                x_half = x1 + (half_max - y1) / slope
                fwhm = 2 * x_half / 60
                label += rf", FWHM $\approx$ {fwhm:.2f}"
        ax.semilogy(bin_centers / 60, avg, color=color, label=label, **plot_kwargs)
        ax.semilogy(-bin_centers / 60, avg, color=color, **plot_kwargs)
    else:
        label = custom_label or r"$\phi=90^\circ$ cut"
        ax.semilogy(theta, cut, color=color, label=label, **plot_kwargs)
        ax.semilogy(-theta, cut, color=color, **plot_kwargs)

    if new_fig:
        ax.set_xlim(-max(theta), max(theta))
        ax.set_ylim([10 ** vmin, 1.1])
        ax.set_xlabel(theta_label)
        ax.set_ylabel("Normalized Beam")
        ax.grid(True)
    if show_legend:
        ax.legend(loc='lower center', framealpha=1, ncol=2)
    if new_fig:
        return fig, ax
def plot_psf_2d_old(
    AA=None,
    uvs=None,
    n_px_side=1000,
    angle_range_deg=90,
    angle_range_arcmin=None,
    vmin=-8,
    vmax=0,
    dpi=150,
    cmap="inferno",
    scaling='power',
    show_cbar=True,
    ax=None,
):
    """
    Plot 2D synthesized beam in polar or cartesian coordinates.

    Parameters
    ----------
    AA : AntArray, optional
        AntArray instance.
    uvs : ndarray, optional
        UV coordinates.
    n_px_side : int
        Resolution.
    angle_range_deg : float
        FOV in degrees.
    angle_range_arcmin : float, optional
        FOV in arcminutes (overrides degrees).
    vmin, vmax : float
        Dynamic range (log10).
    dpi : int
        Figure resolution.
    cmap : str
        Colormap.
    scaling : {'power', 'linear'}
        Intensity scaling.
    show_cbar : bool
        Show colorbar.
    ax : Axes, optional
        Axis to reuse.

    Returns
    -------
    fig, ax : Figure and axis
    """
    if AA is None and uvs is None:
        raise ValueError("Either AA or uvs must be provided.")
    if uvs is None:
        uvs = antpos_to_uv(AA.antpos) / AA.ref_wl

    fov = angle_range_arcmin / 60 if angle_range_arcmin else angle_range_deg
    psf, sin_theta = compute_beam_nufft(uvs, n_px_side=n_px_side, fov_range=fov,dimensions=2)
    if scaling == 'power':
        psf **= 2

    theta = np.arcsin(np.clip(sin_theta, -1, 1)) * (180 / np.pi)
    if angle_range_arcmin:
        theta *= 60
        label = "arcmin"
    else:
        label = "deg"

    THETA_X, THETA_Y = np.meshgrid(theta, theta)
    R = np.sqrt(THETA_X**2 + THETA_Y**2)
    ANGLE = np.arctan2(THETA_Y, THETA_X)

    use_polar = (angle_range_arcmin and angle_range_arcmin > 600) or (not angle_range_arcmin and angle_range_deg > 10)

    if ax is None:
        fig = plt.figure(figsize=(6, 5), dpi=dpi)
        ax = fig.add_subplot(111, projection="polar" if use_polar else None)
    else:
        fig = ax.get_figure()
        if (ax.name == "polar") != use_polar:
            ss = ax.get_subplotspec()
            ax.remove()
            ax = fig.add_subplot(ss, projection="polar" if use_polar else None)

    if use_polar:
        im = ax.pcolormesh(
            ANGLE,
            R,
            psf,
            cmap=cmap,
            norm=colors.LogNorm(vmin=10**vmin, vmax=10**vmax),
            shading="auto",
            rasterized=True,
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, angle_range_arcmin or angle_range_deg)
        ax.set_yticks([])
        ax.grid(False)
    else:
        im = ax.pcolormesh(
            THETA_X,
            THETA_Y,
            psf,
            cmap=cmap,
            norm=colors.LogNorm(vmin=10**vmin, vmax=10**vmax),
            shading="auto",
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(rf"$\theta_x$ [{label}]")
        ax.set_ylabel(rf"$\theta_y$ [{label}]")

    if show_cbar:
        fig.colorbar(im, ax=ax, label="Normalized Beam (log scale)")
    plt.tight_layout()
    return fig, ax


def compute_beam_nufft(uvs: np.ndarray, n_px_side=100, fov_range=90, angle_units='deg', clip = False):
    """
    Compute synthesized beam from uv points using NUFFT.

    Parameters
    ----------
    uvs : ndarray
        UV coordinates (wavelengths), shape (n, 2).
    n_px_side : int
        Image resolution (pixels per side).
    fov_range : float
        Half-field of view in degrees.
    angle_units : str
        Units for fov_range ('deg' or 'arcmin').

    Returns
    -------
    psf : 2D array
        Power beam (normalized).
    sin_theta : 1D array
        Angular grid in sin(theta) coordinates (can extend beyond [-1,1]).
    """

    if angle_units == 'deg':
        fov_rad = fov_range * np.pi/180
    elif angle_units == 'arcmin':
        fov_rad = fov_range / 60 * np.pi/180
    
    # For fov_range <= 90°, this gives sin(theta) which is <= 1
    # For fov_range > 90°, this extends beyond physical sin(theta) limits
    # We use the fov_range directly to scale the coordinate system
    if fov_range <= 90:
        max_sin_theta = np.sin(fov_rad)
    else:
        # Linear scaling: at 90° we have sin(theta)=1, at 180° we want sin(theta)=2
        max_sin_theta = fov_range / 90.0

    uvs = np.vstack([uvs, -uvs])

    # This scaling now allows the coordinate system to extend beyond [-1,1]
    B = n_px_side / 4 / (max_sin_theta + 0.05)

    u_scaled = np.pi * uvs[:, 0] / B
    v_scaled = np.pi * uvs[:, 1] / B
    c = np.ones(len(u_scaled), dtype=np.complex128)
    n_modes = int(n_px_side + 1)
    kx = np.arange(-n_modes//2, n_modes//2)
    x = np.ascontiguousarray(u_scaled)
    y = np.ascontiguousarray(v_scaled)
    eps = 1e-12

    image = nufft2d1(x=x,
        y=y,
        c=c,
        n_modes=(n_modes, n_modes),
        eps=eps,
        isign=1)

    psf = np.abs(image.reshape((n_modes, n_modes))) ** 2
    psf /= psf.max()

    # Extended sin_theta coordinate system
    sin_theta = kx / (2 * B)
    
    # Clip sin_theta to [-1, 1] if requested
    if clip:
        sin_theta = np.clip(sin_theta, -1, 1)  # This line is removed

    return psf, sin_theta




def plot_psf_2d(
    AA=None,
    uvs=None,
    n_px_side=1000,
    angle_range_deg=90,
    angle_range_arcmin=None,
    vmin=-8,
    vmax=0,
    dpi=150,
    cmap="inferno",
    scaling='power',
    show_cbar=True,
    ax=None,
):
    """
    Plot 2D synthesized beam in polar or cartesian coordinates.
    
    Parameters
    ----------
    AA : AntArray, optional
        AntArray instance.
    uvs : ndarray, optional
        UV coordinates.
    n_px_side : int
        Resolution.
    angle_range_deg : float
        FOV in degrees.
    angle_range_arcmin : float, optional
        FOV in arcminutes (overrides degrees).
    vmin, vmax : float
        Dynamic range (log10).
    dpi : int
        Figure resolution.
    cmap : str
        Colormap.
    scaling : {'power', 'linear'}
        Intensity scaling.
    show_cbar : bool
        Show colorbar.
    ax : Axes, optional
        Axis to reuse.
    
    Returns
    -------
    fig, ax : Figure and axis
    """
    if AA is None and uvs is None:
        raise ValueError("Either AA or uvs must be provided.")
    
    if uvs is None:
        uvs = antpos_to_uv(AA.antpos) / AA.ref_wl
    
    if angle_range_arcmin is None:
        fov = angle_range_deg
        angle_units = 'deg'
    else:
        fov = angle_range_arcmin / 60
        angle_units = 'arcmin'
    
    # For extended ranges (> 90 deg), don't clip sin_theta
    # For physical ranges (<= 90 deg), use clipping for backward compatibility
    use_clipping = fov <= 90
    
    psf, sin_theta = compute_beam_nufft(uvs, n_px_side=n_px_side, fov_range=fov, 
                                       angle_units=angle_units, clip=use_clipping)
    
    # Handle extended range beyond physical sin(theta) limits
    if fov <= 90:
        # Physical range: use actual arcsin
        theta = np.degrees(np.arcsin(np.clip(sin_theta, -1, 1)))
    else:
        # Extended range: linear mapping from sin_theta to angle
        # sin_theta ranges from -fov/90 to +fov/90
        # theta should range from -fov to +fov
        theta = sin_theta * (90.0 / 1.0)  # Convert from "effective sin(theta)" to degrees
    
    if angle_units == 'arcmin':
        theta *= 60
        label = "arcmin"
    else:
        label = "deg"
    
    THETA_X, THETA_Y = np.meshgrid(theta, theta)
    R = np.sqrt(THETA_X**2 + THETA_Y**2)
    ANGLE = np.arctan2(THETA_Y, THETA_X)
    
    # For extended ranges beyond 90 deg, force Cartesian coordinates
    # Polar coordinates don't make physical sense beyond the sky
    if fov > 90:
        use_polar = False
    else:
        # Original logic for physical ranges
        use_polar = (angle_units == 'arcmin' and angle_range_arcmin > 600) or \
                   (angle_units == 'deg' and angle_range_deg > 10)
    
    if ax is None:
        fig = plt.figure(figsize=(6, 5), dpi=dpi)
        ax = fig.add_subplot(111, projection="polar" if use_polar else None)
    else:
        fig = ax.get_figure()
        if (ax.name == "polar") != use_polar:
            ss = ax.get_subplotspec()
            ax.remove()
            ax = fig.add_subplot(ss, projection="polar" if use_polar else None)
    
    if use_polar:
        im = ax.pcolormesh(ANGLE, R, psf[::1,::-1],
                          cmap=cmap, norm=colors.LogNorm(vmax=1, vmin=10**vmin), 
                          shading="auto", rasterized=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks([])
        ax.grid(color='k', alpha=0.2)
        ax.set_rlim(0, fov)
    else:
        im = ax.pcolormesh(theta, theta, psf.T[::1,::-1], 
                          cmap=cmap, norm=colors.LogNorm(vmax=1, vmin=10**vmin), 
                          shading="auto", rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel(rf"$\theta_x$ [{label}]")
        ax.set_ylabel(rf"$\theta_y$ [{label}]")
        ax.grid(color='k', alpha=0.2)
        
        # For extended ranges, add a visual indicator of the physical sky boundary
        if fov > 90:
            # Draw a circle at the 90 deg boundary (physical sky limit)
            sky_limit = 90
            if angle_units == 'arcmin':
                sky_limit *= 60
            circle = plt.Circle((0, 0), sky_limit, fill=False, color='white', 
                              linestyle='--', linewidth=2, alpha=0.7)
            ax.add_patch(circle)
            ax.text(0, sky_limit + 0.05 * fov, 'Physical Sky Limit', 
                   ha='center', va='bottom', color='white', fontsize=10, alpha=0.8)
    
    if show_cbar:
        fig.colorbar(im, ax=ax, label="Normalized Beam (log scale)")
    
    plt.tight_layout()
    return fig, ax


def plot_psf_1d(
    AA=None,
    uvs=None,
    n_px_side=1000,
    angle_range_deg=90,
    angle_range_arcmin=None,
    dpi=150,
    ax=None,
    **plot_kwargs,
):
    """
    Plot 1D synthesized beam.
    
    Parameters
    ----------
    AA : AntArray, optional
        AntArray instance.
    uvs : ndarray, optional
        UV coordinates.
    n_px_side : int
        Resolution.
    angle_range_deg : float
        FOV in degrees.
    angle_range_arcmin : float, optional
        FOV in arcminutes (overrides degrees).
    dpi : int
        Figure resolution.
    ax : Axes, optional
        Axis to reuse.
    plot_kwargs : dict
        Extra arguments to pass to plot.
    
    Returns
    -------
    fig, ax : Figure and axis
    """
    if AA is None and uvs is None:
        raise ValueError("Either AA or uvs must be provided.")
    
    if uvs is None:
        uvs = antpos_to_uv(AA.antpos) / AA.ref_wl
    
    if angle_range_arcmin is None:
        fov = angle_range_deg
        angle_units = 'deg'
    else:
        fov = angle_range_arcmin
        angle_units = 'arcmin'
    
    psf, sin_theta = compute_beam_nufft(uvs, n_px_side=n_px_side, fov_range=fov, angle_units=angle_units)
    
    # Handle extended range beyond physical sin(theta) limits
    if fov <= 90:
        # Physical range: use actual arcsin
        theta = np.degrees(np.arcsin(np.clip(sin_theta, -1, 1)))
    else:
        # Extended range: linear mapping from sin_theta to angle
        # sin_theta ranges from -fov/90 to +fov/90
        # theta should range from -fov to +fov
        theta = sin_theta * (90.0 / 1.0)  # Convert from "effective sin(theta)" to degrees
    
    if angle_units == 'arcmin':
        theta *= 60
        label = "arcmin"
    else:
        label = "deg"
    
    THETA_X, THETA_Y = np.meshgrid(theta, theta)
    R = np.sqrt(THETA_X**2 + THETA_Y**2)
    ANGLE = np.arctan2(THETA_Y, THETA_X)
    
    if ax is None:
        fig = plt.figure(figsize=(6, 5), dpi=dpi)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    

    to_plot = psf[psf.shape[0]//2,:]




    ax.semilogy(theta, to_plot, **plot_kwargs)
    ax.set_xlabel(rf"$\theta$ [{label}]")
    ax.grid(color='k', alpha=0.2)
    plt.tight_layout()
    
    return fig, ax
