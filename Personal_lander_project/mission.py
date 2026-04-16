"""
Personal Lunar Lander Project – entry point.

Run from the project root:
    python main.py

Sequence
--------
1. Simulate terminal descent (last ~500 m) with PD controller
2. Show 2D diagnostic plots  (altitude, velocity, mass, controller signals)
3. Compute parking orbit + powered-descent arc for context
4. Show full-mission globe view  (Moon + orbit + arc + terminal phase)
5. Show close-up 3D terminal descent animation

Note on PyVista windows
-----------------------
Each 3D viewer is launched in its own subprocess. This avoids the VTK
OpenGL shader errors that appear when a window is closed and the GPU
context is destroyed — each process gets a fresh context.
"""

import os
import sys
import numpy as np
import yaml
import multiprocessing as mp

from physics.descent_controller import PDDescentController
from physics.orbital_mechanics import circular_orbit, descent_arc
from simulation.integrator import run
from simulation.mission_plots import plot_trajectory, plot_controller_log


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load(path):
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)


ROOT    = os.path.dirname(os.path.abspath(__file__))
cfg_dir = os.path.join(ROOT, 'config')


def _load_configs():
    moon    = _load(os.path.join(cfg_dir, 'moon.yaml'))
    lander  = _load(os.path.join(cfg_dir, 'lander.yaml'))
    sim     = _load(os.path.join(cfg_dir, 'sim.yaml'))
    vehicle = {
        'dry_mass_kg':        lander['dry_mass_kg'],
        'propellant_mass_kg': lander['propellant_mass_kg'],
        'thrust_max_N':       float(lander['engine']['thrust_max_N']),
        'thrust_min_N':       float(lander['engine']['thrust_min_N']),
        'specific_impulse_s': float(lander['engine']['specific_impulse_s']),
    }
    return moon, vehicle, sim


# ---------------------------------------------------------------------------
# Subprocess targets – each runs in its own process with a clean OpenGL ctx
# ---------------------------------------------------------------------------

def _launch_globe(orbit_xyz, arc_xyz, terminal_xyz, moon_params, texture_path):
    """Runs inside a child process."""
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    from simulation.mission_overview import run_globe_viewer
    run_globe_viewer(orbit_xyz, arc_xyz, terminal_xyz, moon_params,
                     texture_path=texture_path)


def _launch_descent(traj_dict, moon_radius_m, telemetry_arrays, speedup):
    """Runs inside a child process."""
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    import pandas as pd
    from simulation.descent_viewer import run_descent_viewer

    traj = pd.DataFrame(traj_dict)

    # Reconstruct a minimal log-like object from the raw arrays
    class _Log:
        def as_arrays(self):
            return telemetry_arrays

    run_descent_viewer(traj, moon_radius_m=moon_radius_m,
                       controller_log=_Log(), speedup=speedup)


def _run_in_process(target, args):
    """Spawn target(*args) in a child process and wait for it."""
    p = mp.Process(target=target, args=args)
    p.start()
    p.join()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    moon_params, vehicle_params, sim_cfg = _load_configs()
    R_moon = float(moon_params['radius_m'])

    # -----------------------------------------------------------------------
    # 1. Parking orbit  (computed first so we can derive terminal IC from it)
    # -----------------------------------------------------------------------
    print("Computing orbital phase...")

    orbit_xyz, orbit_meta = circular_orbit(
        moon_params,
        altitude_m=100_000,
        inclination_rad=0.45,
        raan_rad=0.6,
        n_revolutions=1.5,
    )

    print(f"  Orbit altitude : {orbit_meta['altitude_m']/1000:.0f} km")
    print(f"  Orbital speed  : {orbit_meta['speed_ms']:.1f} m/s")
    print(f"  Orbital period : {orbit_meta['period_s']/60:.1f} min")

    # Deorbit burn point → derive lat/lon for terminal start
    exit_pt  = orbit_xyz[-1]
    r_exit   = np.linalg.norm(exit_pt)
    lat_exit = float(np.arcsin(np.clip(exit_pt[2] / r_exit, -1, 1)))
    lon_exit = float(np.arctan2(exit_pt[1], exit_pt[0]))

    # Orbital velocity direction at exit → decompose into moon-fixed frame
    tan_vec  = orbit_xyz[-1] - orbit_xyz[-2]
    tan_vec /= np.linalg.norm(tan_vec)
    v_circ   = orbit_meta['speed_ms']

    e_east  = np.array([-np.sin(lon_exit),
                         np.cos(lon_exit),
                         0.0])
    e_north = np.array([-np.sin(lat_exit) * np.cos(lon_exit),
                        -np.sin(lat_exit) * np.sin(lon_exit),
                         np.cos(lat_exit)])

    v_east_orb  = float(np.dot(tan_vec * v_circ, e_east))
    v_north_orb = float(np.dot(tan_vec * v_circ, e_north))

    # Simulation starts at the orbit exit with full orbital velocity.
    # The controller fires and the physics determine the landing site.
    terminal_ic = {
        'altitude_m':    100_000.0,
        'latitude_rad':  lat_exit,
        'longitude_rad': lon_exit,
        'v_radial_ms':   0.0,
        'v_east_ms':     v_east_orb,
        'v_north_ms':    v_north_orb,
    }

    print(f"  Deorbit burn at: lat={np.degrees(lat_exit):.2f}°  "
          f"lon={np.degrees(lon_exit):.2f}°")

    # -----------------------------------------------------------------------
    # 2. Full descent simulation (from orbit to touchdown)
    # -----------------------------------------------------------------------
    controller = PDDescentController(moon_params=moon_params,
                                     vehicle_params=vehicle_params)

    print("Running descent simulation (from 100 km orbit)...")
    traj = run(thrust_func=controller.compute_thrust,
               moon_params=moon_params,
               vehicle_params=vehicle_params,
               sim_cfg=sim_cfg,
               ic_override=terminal_ic)

    final = traj.iloc[-1]
    print(
        f"Touchdown  t = {final['time']:.1f} s  |"
        f"  h = {final['altitude']:.1f} m  |"
        f"  Vvert = {final['v_radial']:.2f} m/s  |"
        f"  lat = {np.degrees(final['latitude']):.2f}°  "
        f"lon = {np.degrees(final['longitude']):.2f}°"
    )

    # -----------------------------------------------------------------------
    # 3. 2-D diagnostic plots
    # -----------------------------------------------------------------------
    plot_trajectory(traj)
    plot_controller_log(controller.log)

    # -----------------------------------------------------------------------
    # 4. Convert full trajectory to Cartesian for the globe viewer
    #    No separate arc needed — the simulation IS the descent path
    # -----------------------------------------------------------------------
    r_t   = traj['r'].values
    lat_t = traj['latitude'].values
    lon_t = traj['longitude'].values
    descent_xyz = np.column_stack([
        r_t * np.cos(lat_t) * np.cos(lon_t),
        r_t * np.cos(lat_t) * np.sin(lon_t),
        r_t * np.sin(lat_t),
    ])

    # -----------------------------------------------------------------------
    # 5. Launch 3D viewers sequentially
    #    1) Globe first — user explores the full mission overview, then closes it
    #    2) Close-up descent viewer — shows last 300 m, closing it exits everything
    # -----------------------------------------------------------------------
    texture_path = os.path.join(ROOT, 'assets', 'moon_texture.jpg')
    telemetry_arrays = controller.log.as_arrays()

    # Filter trajectory to last 300 m for the close-up viewer
    traj_close = traj[traj['altitude'] <= 300.0].reset_index(drop=True)
    if len(traj_close) < 2:
        print("Warning: fewer than 2 points below 300 m — using full trajectory.")
        traj_close = traj

    # --- Globe viewer (blocks until user closes the window) ---
    print("Launching globe viewer  (close window to continue to descent animation)...")
    globe_proc = mp.Process(
        target=_launch_globe,
        args=(orbit_xyz, np.array([]), descent_xyz, moon_params, texture_path),
    )
    globe_proc.start()
    globe_proc.join()   # wait for user to close the globe

    # --- Close-up descent viewer ---
    print("Launching terminal descent viewer  (close window to exit)...")
    descent_proc = mp.Process(
        target=_launch_descent,
        args=(traj_close.to_dict(orient='list'), R_moon, telemetry_arrays, 5.0),
    )
    descent_proc.start()
    descent_proc.join()


if __name__ == '__main__':
    # Required on Windows so spawned processes don't re-run this module
    mp.freeze_support()
    main()
