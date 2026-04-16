"""
Circular parking orbit generator (Keplerian, two-body).

Computes positions along a circular orbit at a given altitude above the Moon.
The orbital plane is defined by an inclination angle and a longitude of the
ascending node (RAAN), which can be tuned so the orbit passes over a chosen
landing site.

All positions are returned in Moon-centred Cartesian coordinates [m].
"""

import numpy as np


def circular_orbit(moon_params, altitude_m, inclination_rad=0.4,
                   raan_rad=0.0, n_revolutions=1.5, n_points=1500):
    """
    Generate Cartesian positions for a circular orbit.

    Parameters
    ----------
    moon_params : dict
        Must contain 'grav_param_m3s2' and 'radius_m'.
    altitude_m : float
        Orbit altitude above the lunar surface [m].
    inclination_rad : float
        Orbital inclination with respect to the lunar equator [rad].
    raan_rad : float
        Right Ascension of the Ascending Node [rad].
    n_revolutions : float
        Number of full revolutions to generate.
    n_points : int
        Number of discrete points along the orbit.

    Returns
    -------
    positions : ndarray, shape (n_points, 3)
        Cartesian [x, y, z] positions [m].
    metadata : dict
        Orbital parameters: radius, speed, period.
    """
    GM     = float(moon_params['grav_param_m3s2'])
    R_moon = float(moon_params['radius_m'])

    r_orbit = R_moon + altitude_m
    v_circ  = np.sqrt(GM / r_orbit)
    period  = 2.0 * np.pi * np.sqrt(r_orbit**3 / GM)

    # True anomaly parameter
    nu = np.linspace(0.0, n_revolutions * 2.0 * np.pi, n_points)

    # Positions in the perifocal frame (orbit lies in x-y plane)
    x_peri = r_orbit * np.cos(nu)
    y_peri = r_orbit * np.sin(nu)
    z_peri = np.zeros(n_points)

    # Rotation 1: inclination (rotate around x-axis by i)
    ci, si = np.cos(inclination_rad), np.sin(inclination_rad)
    x1 =  x_peri
    y1 =  y_peri * ci - z_peri * si
    z1 =  y_peri * si + z_peri * ci

    # Rotation 2: RAAN (rotate around z-axis by Omega)
    co, so = np.cos(raan_rad), np.sin(raan_rad)
    x2 =  x1 * co - y1 * so
    y2 =  x1 * so + y1 * co
    z2 =  z1

    positions = np.column_stack([x2, y2, z2])

    metadata = {
        'radius_m':  r_orbit,
        'speed_ms':  v_circ,
        'period_s':  period,
        'altitude_m': altitude_m,
    }
    return positions, metadata


def descent_arc(orbit_exit_xyz, terminal_start_xyz, orbit_tangent=None, n_points=400):
    """
    Generate a smooth powered-descent arc from the deorbit burn point
    to the start of the terminal descent phase.

    Uses spherical coordinate interpolation so the arc always follows
    the Moon surface (no loops or outward excursions).
    The altitude profile uses a cubic ease-in so the descent starts
    gently (nearly horizontal) and steepens toward the surface.

    Parameters
    ----------
    orbit_exit_xyz : array-like (3,)   Cartesian deorbit position [m].
    terminal_start_xyz : array-like (3,)  Cartesian terminal start [m].
    orbit_tangent : ignored (kept for API compatibility).
    n_points : int

    Returns
    -------
    ndarray, shape (n_points, 3)
    """
    p0 = np.asarray(orbit_exit_xyz,      dtype=float)
    p1 = np.asarray(terminal_start_xyz,  dtype=float)

    # Convert to spherical
    def to_sph(p):
        r   = np.linalg.norm(p)
        lat = np.arcsin(np.clip(p[2] / r, -1, 1))
        lon = np.arctan2(p[1], p[0])
        return r, lat, lon

    r0, lat0, lon0 = to_sph(p0)
    r1, lat1, lon1 = to_sph(p1)

    t = np.linspace(0.0, 1.0, n_points)

    # Cubic ease-in for altitude: slow descent at start, steep near surface
    t_r   = 3 * t**2 - 2 * t**3          # smooth-step (S-curve)
    r_arc = r0 + (r1 - r0) * t_r

    # Linear interpolation for lat/lon (great-circle approximation)
    lat_arc = lat0 + (lat1 - lat0) * t
    lon_arc = lon0 + (lon1 - lon0) * t

    # Back to Cartesian
    x = r_arc * np.cos(lat_arc) * np.cos(lon_arc)
    y = r_arc * np.cos(lat_arc) * np.sin(lon_arc)
    z = r_arc * np.sin(lat_arc)

    return np.column_stack([x, y, z])
