"""
Simulation runner.

Wraps scipy's solve_ivp to integrate the equations of motion forward in time,
stopping automatically when the lander touches down.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from physics.lunar_eom import equations_of_motion


def build_initial_state(moon_params, sim_cfg):
    """Assemble the 7-element initial state vector from config dicts."""
    ic = sim_cfg['initial_conditions']
    r0 = moon_params['radius_m'] + ic['altitude_m']
    return [
        r0,
        ic['latitude_rad'],
        ic['longitude_rad'],
        ic['v_radial_ms'],
        ic['v_east_ms'],
        ic['v_north_ms'],
        moon_params.get('initial_mass_kg', 1200.0),  # pulled from lander cfg at call site
    ]


def _surface_event(moon_radius):
    """Returns a scipy event that fires when r == R_moon (touchdown)."""
    def event(t, y):
        return y[0] - moon_radius
    event.terminal  = True
    event.direction = -1   # only trigger on descent
    return event


def run(thrust_func, moon_params, vehicle_params, sim_cfg, ic_override=None):
    """
    Integrate the lander EOM and return a tidy DataFrame of the trajectory.

    Parameters
    ----------
    thrust_func : callable
        Signature: (t, state) -> float  [Newtons]
    moon_params, vehicle_params, sim_cfg : dict
        Loaded from the YAML config files.

    Returns
    -------
    pd.DataFrame
        Columns: time, r, latitude, longitude, v_radial, v_east, v_north, mass, altitude
    """
    cfg_int = sim_cfg['integrator']
    t_end   = cfg_int['t_end_s']
    dt_out  = cfg_int['dt_output_s']

    # Merge lander mass into moon_params for convenience
    total_mass = (vehicle_params['dry_mass_kg']
                  + vehicle_params['propellant_mass_kg'])

    y0 = build_initial_state(moon_params, sim_cfg)
    y0[6] = total_mass

    # Allow caller to override initial conditions (e.g. derived from orbit)
    if ic_override is not None:
        R = float(moon_params['radius_m'])
        y0[0] = R + float(ic_override['altitude_m'])
        y0[1] = float(ic_override['latitude_rad'])
        y0[2] = float(ic_override['longitude_rad'])
        y0[3] = float(ic_override['v_radial_ms'])
        y0[4] = float(ic_override['v_east_ms'])
        y0[5] = float(ic_override['v_north_ms'])

    t_eval = np.arange(0.0, t_end, dt_out)

    def rhs(t, y):
        return equations_of_motion(t, y, thrust_func, moon_params, vehicle_params)

    result = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        method=cfg_int.get('method', 'RK45'),
        t_eval=t_eval,
        events=_surface_event(moon_params['radius_m']),
        max_step=cfg_int.get('max_step_s', np.inf),
        dense_output=False,
    )

    r, lat, lon, w, u, v, mass = result.y
    altitude = r - moon_params['radius_m']

    df = pd.DataFrame({
        'time':      result.t,
        'r':         r,
        'latitude':  lat,
        'longitude': lon,
        'v_radial':  w,
        'v_east':    u,
        'v_north':   v,
        'mass':      mass,
        'altitude':  altitude,
    })

    return df
