"""
Visualisation module.

Provides:
  - plot_trajectory(df)          : static 2-D overview of the descent
  - plot_controller_log(log)     : controller internal signals over time
  - animate_descent(df)          : 3-D animated descent in Cartesian space
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# Coordinate helper
# ---------------------------------------------------------------------------

def spherical_to_cartesian(r, lat, lon):
    """
    Convert moon-fixed spherical (r, lat, lon) to Cartesian (x, y, z).

    Parameters
    ----------
    r, lat, lon : float or ndarray
        Radial distance [m], latitude [rad], longitude [rad].

    Returns
    -------
    x, y, z : ndarray
    """
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


# ---------------------------------------------------------------------------
# Static 2-D trajectory overview
# ---------------------------------------------------------------------------

def plot_trajectory(df, title="Lunar Descent Trajectory"):
    """
    Four-panel overview:  altitude, vertical speed, horizontal speed, mass.

    Parameters
    ----------
    df : pd.DataFrame
        Output of simulation.runner.run()
    """
    t   = df['time'].values
    alt = df['altitude'].values
    w   = df['v_radial'].values
    u   = df['v_east'].values
    v   = df['v_north'].values
    m   = df['mass'].values
    v_h = np.sqrt(u**2 + v**2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=13)

    axes[0, 0].plot(t, alt, color='steelblue')
    axes[0, 0].set_ylabel('Altitude [m]')
    axes[0, 0].set_title('Altitude vs Time')
    axes[0, 0].grid(True, alpha=0.35)

    axes[0, 1].plot(t, w, color='tomato', label='vertical (w)')
    axes[0, 1].plot(t, v_h, '--', color='dimgray', alpha=0.7, label='horizontal')
    axes[0, 1].set_ylabel('Speed [m/s]')
    axes[0, 1].set_title('Velocity Components')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.35)

    axes[1, 0].plot(t, u, color='darkorange', label='east (u)')
    axes[1, 0].plot(t, v, color='mediumpurple', label='north (v)')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Speed [m/s]')
    axes[1, 0].set_title('Horizontal Velocity Breakdown')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.35)

    axes[1, 1].plot(t, m, color='seagreen')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Mass [kg]')
    axes[1, 1].set_title('Propellant Consumption')
    axes[1, 1].grid(True, alpha=0.35)

    plt.tight_layout()
    plt.show(block=True)


# ---------------------------------------------------------------------------
# Controller telemetry plots
# ---------------------------------------------------------------------------

def plot_controller_log(log, title="Controller Telemetry"):
    """
    Plot internal controller signals from a TelemetryLog.

    Parameters
    ----------
    log : physics.controller.TelemetryLog
    """
    data = log.as_arrays()
    t    = data['times']

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(title, fontsize=13)

    axes[0].plot(t, data['alt_errors'], color='steelblue')
    axes[0].set_ylabel('Altitude error [m]')
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(t, data['vel_errors'], color='tomato')
    axes[1].set_ylabel('Velocity error [m/s]')
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(t, data['accel_cmds'], color='darkorange')
    axes[2].set_ylabel('Commanded accel [m/s²]')
    axes[2].grid(True, alpha=0.35)

    axes[3].plot(t, data['thrust_cmds'], color='dimgray')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Thrust [N]')
    axes[3].grid(True, alpha=0.35)

    plt.tight_layout()
    plt.show(block=True)


# ---------------------------------------------------------------------------
# 3-D animated descent
# ---------------------------------------------------------------------------

def animate_descent(df, moon_radius_m, interval_ms=15):
    """
    Animate the lander's descent in 3-D Cartesian space.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory from simulation.runner.run().
    moon_radius_m : float
        Used to compute altitude for the info overlay.
    interval_ms : int
        Milliseconds between animation frames.

    Returns
    -------
    anim : FuncAnimation
        Keep a reference to prevent garbage collection.
    """
    x, y, z = spherical_to_cartesian(
        df['r'].values,
        df['latitude'].values,
        df['longitude'].values,
    )
    t   = df['time'].values
    alt = df['altitude'].values

    # Approximate speed from finite differences on Cartesian positions
    xyz  = np.column_stack([x, y, z])
    dt   = np.diff(t)
    dxyz = np.diff(xyz, axis=0)
    spd  = np.linalg.norm(dxyz / dt[:, None], axis=1)
    spd  = np.append(spd, spd[-1])   # repeat last value to match length

    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # Static path
    ax.plot(x, y, z, color='lightgray', linewidth=0.8, alpha=0.7)

    # Lander marker
    lander, = ax.plot([x[0]], [y[0]], [z[0]],
                      'o', color='crimson', markersize=10, label='Lander')

    info = ax.text2D(
        0.02, 0.96, "",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75),
    )

    # Mark start / end
    ax.plot(*xyz[0],  'x', color='crimson', markersize=12, label='Start')
    ax.plot(*xyz[-1], 'x', color='navy',    markersize=12, label='Touchdown')

    # Axes limits centred on start with generous padding
    pad = max(500.0, alt[0] * 1.5)
    ax.set_xlim(x[0] - pad, x[0] + pad)
    ax.set_ylim(y[0] - pad, y[0] + pad)
    ax.set_zlim(z[0] - pad, z[0] + pad)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Lunar Descent – 3D Animation')
    ax.legend(fontsize=8)

    def _update(i):
        lander.set_data([x[i]], [y[i]])
        lander.set_3d_properties([z[i]])
        info.set_text(
            f"t = {t[i]:7.2f} s\n"
            f"h = {alt[i]:7.1f} m\n"
            f"v = {spd[i]:7.2f} m/s"
        )
        return lander, info

    anim = FuncAnimation(
        fig, _update,
        frames=len(t),
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()
    return anim
