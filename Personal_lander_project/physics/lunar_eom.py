"""
Equations of motion for a powered descent vehicle in a rotating spherical frame.

Reference frame
---------------
The frame is fixed to the rotating Moon. The state is expressed in spherical
coordinates (r, lat, lon) with velocity components:
    w  – radial (positive outward)
    u  – eastward (longitude direction)
    v  – northward (latitude direction)

The acceleration terms come from Newton's second law written in a non-inertial
rotating frame and include:
    - gravity (central field, μ/r²)
    - thrust (3-component vector: radial, eastward, northward)
    - Coriolis acceleration (from frame rotation Ω)
    - centrifugal acceleration (from frame rotation Ω)
    - curvature / transport terms (spherical geometry)

Thrust interface
----------------
thrust_func(t, state) may return:
    - a scalar float  → applied in the radial direction only
    - a 3-element array-like [T_r, T_e, T_n]  → full 3-axis thrust

State vector layout
-------------------
    idx 0  r      radial distance from Moon centre [m]
    idx 1  lat    geocentric latitude [rad]
    idx 2  lon    geocentric longitude [rad]
    idx 3  w      radial velocity [m/s]
    idx 4  u      eastward velocity [m/s]
    idx 5  v      northward velocity [m/s]
    idx 6  mass   current vehicle mass [kg]

References
----------
Battin, R. H. – "An Introduction to the Mathematics and Methods of Astrodynamics"
Wie, B.       – "Space Vehicle Dynamics and Control"
"""

import numpy as np


def equations_of_motion(t, state, thrust_func, moon_params, vehicle_params):
    """
    Compute the time derivative of the state vector.

    Parameters
    ----------
    t : float
        Current time [s].
    state : array-like, length 7
        Current state [r, lat, lon, w, u, v, mass].
    thrust_func : callable
        thrust_func(t, state) -> float or array-like of 3
        Scalar → radial only.  3-vector → [T_radial, T_east, T_north] [N].
    moon_params : dict
        Must contain keys: 'grav_param_m3s2', 'spin_rate_rads'.
    vehicle_params : dict
        Must contain keys: 'specific_impulse_s', 'thrust_max_N'.

    Returns
    -------
    list of float, length 7
        [dr, dlat, dlon, dw, du, dv, dmass]
    """
    r, lat, lon, w, u, v, mass = state

    # Guard: frozen state when propellant is gone or vehicle is on the surface
    if mass <= 0.0 or r <= moon_params['radius_m']:
        return [0.0] * 7

    GM    = float(moon_params['grav_param_m3s2'])
    Omega = float(moon_params['spin_rate_rads'])
    Isp   = float(vehicle_params['specific_impulse_s'])
    T_max = float(vehicle_params['thrust_max_N'])
    G0    = 9.80665  # standard gravity [m/s²]

    # ------------------------------------------------------------------
    # Thrust vector: accept scalar or 3-component from the controller
    # ------------------------------------------------------------------
    T_raw = thrust_func(t, state)
    T_arr = np.asarray(T_raw, dtype=float).ravel()

    if T_arr.size >= 3:
        T_r, T_e, T_n = float(T_arr[0]), float(T_arr[1]), float(T_arr[2])
    else:
        T_r, T_e, T_n = float(T_arr[0]), 0.0, 0.0

    # Clip total thrust magnitude to engine limits
    T_mag = float(np.sqrt(T_r**2 + T_e**2 + T_n**2))
    if T_mag > T_max:
        scale = T_max / T_mag
        T_r *= scale; T_e *= scale; T_n *= scale
        T_mag = T_max
    T_mag = max(T_mag, 0.0)

    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    tan_lat = np.tan(lat)

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    dr   = w
    dlat = v / r
    dlon = u / (r * cos_lat)

    # ------------------------------------------------------------------
    # Radial acceleration
    # ------------------------------------------------------------------
    dw = (T_r / mass) \
         - (GM / r**2) \
         + (u**2 + v**2) / r \
         - 2.0 * u * Omega * cos_lat \
         + r * Omega**2 * cos_lat**2

    # ------------------------------------------------------------------
    # Eastward acceleration
    # ------------------------------------------------------------------
    du = (T_e / mass) \
         + (- u * w + u * v * tan_lat) / r \
         - 2.0 * w * Omega * cos_lat \
         + 2.0 * v * Omega * sin_lat

    # ------------------------------------------------------------------
    # Northward acceleration
    # ------------------------------------------------------------------
    dv = (T_n / mass) \
         + (- v * w - u**2 * tan_lat) / r \
         - 2.0 * u * Omega * sin_lat \
         - r * Omega**2 * sin_lat * cos_lat

    # ------------------------------------------------------------------
    # Propellant consumption  (Tsiolkovsky)
    # ------------------------------------------------------------------
    dmass = -T_mag / (Isp * G0)

    return [dr, dlat, dlon, dw, du, dv, dmass]
