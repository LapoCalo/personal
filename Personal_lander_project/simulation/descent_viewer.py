"""
Terminal descent 3D viewer — close-up animated view.

Scene:
  - 10 km x 10 km lunar surface centred on the touchdown point
  - 1 km wireframe grid for scale
  - Green landing target disc
  - Animated lunar lander model (body + legs + nozzle)
  - Growing yellow path trail
  - Live telemetry overlay (time, altitude, speed, T/W, mass)
"""

import sys
import os
import time
import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# Lander geometry helpers
# ---------------------------------------------------------------------------

def _make_lander_mesh(scale=8.0):
    """
    Build a simple lunar-lander shape in local space.

    Local frame: origin at footpad level, +Z pointing up (toward sky).
    The mesh is created once and repositioned every frame via user_matrix.
    """
    parts = []

    # --- Descent stage body (flat cylinder) ---
    body = pv.Cylinder(
        center=(0, 0, scale * 0.45),
        direction=(0, 0, 1),
        radius=scale * 0.42,
        height=scale * 0.7,
        resolution=12,
    )
    parts.append(body)

    # --- Ascent stage (smaller cylinder on top) ---
    top = pv.Cylinder(
        center=(0, 0, scale * 1.1),
        direction=(0, 0, 1),
        radius=scale * 0.22,
        height=scale * 0.45,
        resolution=12,
    )
    parts.append(top)

    # --- Engine nozzle (cone pointing down) ---
    nozzle = pv.Cone(
        center=(0, 0, scale * 0.04),
        direction=(0, 0, -1),
        height=scale * 0.28,
        radius=scale * 0.18,
        resolution=10,
    )
    parts.append(nozzle)

    # --- 4 landing legs ---
    for angle_deg in [45, 135, 225, 315]:
        rad = np.radians(angle_deg)
        cx, cy = np.cos(rad), np.sin(rad)

        p1 = np.array([cx * scale * 0.32, cy * scale * 0.32, scale * 0.18])
        p2 = np.array([cx * scale * 0.88, cy * scale * 0.88, 0.0])
        mid = (p1 + p2) / 2.0
        leg_vec = p2 - p1
        leg_len = float(np.linalg.norm(leg_vec))

        strut = pv.Cylinder(
            center=mid,
            direction=leg_vec / leg_len,
            radius=scale * 0.035,
            height=leg_len,
            resolution=4,
        )
        parts.append(strut)

        pad = pv.Cylinder(
            center=(cx * scale * 0.88, cy * scale * 0.88, -scale * 0.025),
            direction=(0, 0, 1),
            radius=scale * 0.11,
            height=scale * 0.05,
            resolution=8,
        )
        parts.append(pad)

    mesh = parts[0]
    for p in parts[1:]:
        mesh = mesh.merge(p)
    return mesh


def _lander_transform(pos, direction):
    """
    Return a 4×4 numpy matrix that:
      - rotates the local +Z axis to point OPPOSITE to `direction` (thrust vector),
      - translates the origin to `pos`.

    Physics: if thrust points up (T_r > 0), the lander's nozzle must point DOWN
    to exhaust gas downward. So lander axis = -thrust_direction.
    """
    d = np.array(direction, dtype=float)
    d_mag = np.linalg.norm(d)

    if d_mag < 0.1:
        # No thrust: align upright along surface normal
        up = pos / np.linalg.norm(pos)
    else:
        # +Z of lander aligns WITH thrust direction:
        # thrust points away from Moon (decelerating) → lander stands upright
        # nozzle (at local z=0, bottom of mesh) naturally faces the Moon
        up = d / d_mag

    # Build orthonormal frame with up as Z
    ref = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    nozzle_dir = up
    e_x = np.cross(ref, nozzle_dir);  e_x /= np.linalg.norm(e_x)
    e_y = np.cross(nozzle_dir, e_x)

    mat = np.eye(4)
    mat[:3, 0] = e_x
    mat[:3, 1] = e_y
    mat[:3, 2] = nozzle_dir
    mat[:3, 3] = pos
    return mat


# ---------------------------------------------------------------------------
# Coordinate helper
# ---------------------------------------------------------------------------

def _to_cartesian(r, lat, lon):
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def run_descent_viewer(trajectory_df, moon_radius_m,
                       controller_log=None, speedup=5.0):
    """
    Animate the terminal descent phase.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Columns: r, latitude, longitude, altitude, v_radial,
                 v_east, v_north, time, mass.
    moon_radius_m : float
    controller_log : TelemetryLog or None
    speedup : float   Playback speed multiplier.
    """
    # ------------------------------------------------------------------
    # Unpack trajectory
    # ------------------------------------------------------------------
    r   = trajectory_df['r'].values
    lat = trajectory_df['latitude'].values
    lon = trajectory_df['longitude'].values
    alt = trajectory_df['altitude'].values
    w   = trajectory_df['v_radial'].values
    u   = trajectory_df['v_east'].values
    v   = trajectory_df['v_north'].values
    t   = trajectory_df['time'].values
    m   = trajectory_df['mass'].values

    xs, ys, zs  = _to_cartesian(r, lat, lon)
    path_xyz    = np.column_stack([xs, ys, zs])
    speed       = np.sqrt(w**2 + u**2 + v**2)
    dt_sim      = float(t[1] - t[0]) if len(t) > 1 else 0.02

    # Touchdown point and surface normal (constant throughout animation)
    land_pos = path_xyz[-1]
    normal   = land_pos / np.linalg.norm(land_pos)

    # Interpolate thrust components for lander orientation + T/W display
    thrust_data = None
    thrust_components = None  # [T_r, T_e, T_n] interpolators
    if controller_log is not None:
        td = controller_log.as_arrays()
        if len(td['times']) > 0:
            from scipy.interpolate import interp1d
            t_u, idx = np.unique(td['times'], return_index=True)
            T_u = td['thrust_cmds'][idx]
            T_r_u = td['thrust_r'][idx]
            T_e_u = td['thrust_e'][idx]
            T_n_u = td['thrust_n'][idx]
            f_T = interp1d(t_u, T_u, bounds_error=False, fill_value=0.0)
            f_Tr = interp1d(t_u, T_r_u, bounds_error=False, fill_value=0.0)
            f_Te = interp1d(t_u, T_e_u, bounds_error=False, fill_value=0.0)
            f_Tn = interp1d(t_u, T_n_u, bounds_error=False, fill_value=0.0)
            thrust_data = f_T(t)
            thrust_components = (f_Tr, f_Te, f_Tn)

    # ------------------------------------------------------------------
    # Build scene
    # ------------------------------------------------------------------
    plotter = pv.Plotter(title="Terminal Descent – Close-up View")
    plotter.set_background([0.05, 0.05, 0.12])   # dark navy blue

    # 10 km × 10 km surface — warm dark basalt tone
    surface = pv.Plane(
        center=land_pos,
        direction=land_pos,
        i_size=10_000, j_size=10_000,
    )
    plotter.add_mesh(surface, color=[0.22, 0.20, 0.18], label="Surface")

    # 1 km wireframe grid — subtle warm grey
    grid = pv.Plane(
        center=land_pos,
        direction=land_pos,
        i_size=10_000, j_size=10_000,
        i_resolution=10, j_resolution=10,
    )
    plotter.add_mesh(grid, color=[0.38, 0.35, 0.32], style="wireframe", line_width=1)

    # Landing target — smooth cyan circle (spline on the surface plane)
    n_ring  = 256
    r_ring  = 20.0
    _r  = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    _t1 = np.cross(normal, _r);  _t1 /= np.linalg.norm(_t1)
    _t2 = np.cross(normal, _t1)
    angles   = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    ring_pts = np.array([
        land_pos + r_ring * (np.cos(a) * _t1 + np.sin(a) * _t2)
        for a in angles
    ])
    ring_pts = np.vstack([ring_pts, ring_pts[0]])   # close the loop
    plotter.add_mesh(pv.Spline(ring_pts, n_ring + 1),
                     color=[0.0, 0.85, 0.85], line_width=2, label="Target")

    # Lander model — metallic gold tint
    lander_mesh  = _make_lander_mesh(scale=8.0)
    lander_actor = plotter.add_mesh(lander_mesh, color=[0.80, 0.72, 0.45],
                                    smooth_shading=True, label="Lander",
                                    specular=0.6, specular_power=30)
    lander_actor.user_matrix = _lander_transform(path_xyz[0], normal)

    # Path trail — bright orange
    path_actor = plotter.add_mesh(
        pv.MultipleLines(path_xyz[0:2]),
        color=[1.0, 0.55, 0.0], line_width=2.5, label="Path"
    )

    # Telemetry text overlay — courier-style monospace, slightly larger
    telemetry_actor = plotter.add_text(
        "", position="lower_left", font_size=11,
        color=[0.85, 0.95, 1.0],   # light ice blue
        font="courier",
    )

    plotter.add_legend(loc="upper right", size=(0.2, 0.18), bcolor=[0.0, 0.0, 0.0, 0.5])

    # Camera: surface-normal–based so it is never underground
    ref_up  = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e_side  = np.cross(normal, ref_up);  e_side /= np.linalg.norm(e_side)
    cam_pos = tuple(land_pos + normal * 200 + e_side * 500)
    foc_pt  = tuple(land_pos + normal * 20)
    plotter.camera_position = [cam_pos, foc_pt, tuple(normal)]
    plotter.camera.clipping_range = (1, 10_000)

    # ------------------------------------------------------------------
    # Animation loop
    # ------------------------------------------------------------------
    plotter.show(auto_close=False, interactive_update=True)

    real_dt = dt_sim / speedup
    G_MOON  = 1.62

    try:
        for i in range(len(path_xyz)):
            frame_start = time.time()

            if plotter.render_window.GetNeverRendered():
                break

            pos = path_xyz[i]

            # Reposition + reorient lander: orient along thrust vector
            # The thrust is stored in spherical frame [T_r, T_e, T_n].
            # Convert to Cartesian using local basis vectors at current position.
            if thrust_components is not None:
                f_Tr, f_Te, f_Tn = thrust_components
                T_r_f = float(f_Tr(t[i]))
                T_e_f = float(f_Te(t[i]))
                T_n_f = float(f_Tn(t[i]))

                # Local basis vectors at this lat/lon
                _lat, _lon = lat[i], lon[i]
                e_r = np.array([
                    np.cos(_lat) * np.cos(_lon),
                    np.cos(_lat) * np.sin(_lon),
                    np.sin(_lat),
                ])
                e_e = np.array([-np.sin(_lon), np.cos(_lon), 0.0])
                e_n = np.array([
                    -np.sin(_lat) * np.cos(_lon),
                    -np.sin(_lat) * np.sin(_lon),
                     np.cos(_lat),
                ])

                # Thrust in Cartesian world frame
                thrust_vec = T_r_f * e_r + T_e_f * e_e + T_n_f * e_n
            else:
                thrust_vec = normal  # default to radial if no thrust data

            lander_actor.user_matrix = _lander_transform(pos, thrust_vec)

            # Grow path trail
            if i > 1:
                path_actor.mapper.dataset.copy_from(
                    pv.MultipleLines(path_xyz[:i + 1])
                )

            # Telemetry overlay
            tw_str = "0.00"
            if thrust_data is not None and m[i] > 0:
                tw_str = f"{thrust_data[i] / (m[i] * G_MOON):.2f}"

            # Compute horizontal velocity
            v_horiz = float(np.sqrt(u[i]**2 + v[i]**2))

            stats = (
                f"Time: {t[i]:8.2f} s  ({speedup:.1f}x)\n"
                f"Altitude: {alt[i]:8.2f} m\n"
                f"V vertical: {abs(w[i]):8.2f} m/s ↓\n"
                f"V horiz:    {v_horiz:8.2f} m/s →\n"
                f"T/W:        {tw_str}\n"
                f"Mass:       {m[i]:7.1f} kg"
            )
            telemetry_actor.set_text(0, stats)

            plotter.update()

            elapsed = time.time() - frame_start
            wait    = real_dt - elapsed
            if wait > 0:
                time.sleep(wait)

        print("Landing complete.")
        plotter.show()

    except Exception as exc:
        print(f"Viewer stopped: {exc}")
    finally:
        plotter.close()
        os._exit(0)
