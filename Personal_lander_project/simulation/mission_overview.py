"""
Full-mission globe viewer.

Renders the complete mission around the Moon:
  - Moon sphere with real NASA texture (auto-downloaded on first run)
  - Starfield background
  - Circular parking orbit (blue)
  - Powered-descent arc from orbit to simulation start (orange)
  - Simulated terminal descent – last 1000 m (bright yellow)
  - Event markers: deorbit burn, terminal-phase start, touchdown

The three trajectory segments are contiguous:
    orbit end  →  arc start
    arc end    →  terminal_xyz[0]   (exact match, no gap)
    terminal_xyz[-1]  =  touchdown

Texture
-------
On first run the viewer downloads a 1K equirectangular Moon texture
(~2 MB) from NASA's public-domain Visible Earth catalogue and saves it
to assets/moon_texture.jpg. Subsequent runs use the cached file.
If the download fails the viewer falls back to a shaded grey sphere.
"""

import os
import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# Texture management
# ---------------------------------------------------------------------------

_TEXTURE_URL = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/"
    "world.topo.bathy.200412.3x5400x2700.jpg"
)
# This is NASA's "Blue Marble" Earth image — we use it as a stand-in when
# no Moon texture is present, but the preferred file is the Moon texture below.
_MOON_TEXTURE_URL = (
    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/"
    "lroc_color_poles_1k.jpg"
)


def _ensure_texture(texture_path):
    """
    Return a path to a valid texture file, downloading if necessary.
    Returns None if no texture is available.
    """
    if os.path.isfile(texture_path):
        return texture_path

    # Try to download
    try:
        import urllib.request
        os.makedirs(os.path.dirname(texture_path), exist_ok=True)
        print("  Downloading Moon texture from NASA (first run only, ~3 MB)...")
        urllib.request.urlretrieve(_MOON_TEXTURE_URL, texture_path)
        print(f"  Saved to {texture_path}")
        return texture_path
    except Exception as exc:
        print(f"  Texture download failed ({exc}). Using shaded grey sphere.")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _starfield(n=4000, radius=1.0e8):
    phi   = np.random.uniform(0, 2 * np.pi, n)
    theta = np.arccos(np.random.uniform(-1.0, 1.0, n))
    r     = np.random.uniform(0.95, 1.0, n) * radius
    pts   = np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])
    return pv.PolyData(pts)


def _moon_actor(pl, R_moon, texture_path):
    """Add the Moon sphere to the plotter. Returns the actor."""
    sphere = pv.Sphere(radius=R_moon,
                       theta_resolution=128,
                       phi_resolution=128)

    if texture_path and os.path.isfile(texture_path):
        sphere = sphere.texture_map_to_sphere(inplace=False)
        tex    = pv.read_texture(texture_path)
        return pl.add_mesh(sphere, texture=tex, smooth_shading=True,
                           lighting=True)

    # Fallback: pseudo-lighting via scalar field
    sphere['brightness'] = sphere.points[:, 2]
    return pl.add_mesh(sphere, scalars='brightness', cmap='gray',
                       clim=[-R_moon * 1.4, R_moon * 1.4],
                       show_scalar_bar=False, smooth_shading=True)


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def run_globe_viewer(orbit_xyz, arc_xyz, descent_xyz, moon_params,
                     texture_path='assets/moon_texture.jpg'):
    """
    Launch the interactive full-mission globe view.

    Parameters
    ----------
    orbit_xyz   : ndarray (N, 3) – parking orbit [m]
    arc_xyz     : ndarray (M, 3) – intermediate arc (may be empty)
    descent_xyz : ndarray (K, 3) – full simulated descent trajectory [m]
    moon_params : dict           – must contain 'radius_m'
    texture_path : str           – local path; downloaded if absent
    """
    R_moon = float(moon_params['radius_m'])
    resolved = _ensure_texture(texture_path)

    pl = pv.Plotter(title="Lunar Mission – Full Trajectory Overview")
    pl.set_background([0.05, 0.05, 0.12])   # dark navy blue (matches descent viewer)

    pl.add_mesh(_starfield(4000, R_moon * 60), color='white', point_size=1,
                render_points_as_spheres=True, lighting=False)

    _moon_actor(pl, R_moon, resolved)

    # Parking orbit — dodger blue
    pl.add_mesh(pv.Spline(orbit_xyz, len(orbit_xyz)),
                color=[0.12, 0.56, 1.0], line_width=2,
                label='Parking orbit (100 km)')

    # Optional intermediate arc
    if len(arc_xyz) >= 2:
        pl.add_mesh(pv.Spline(arc_xyz, len(arc_xyz)),
                    color=[1.0, 0.55, 0.0], line_width=2,
                    label='Descent arc')

    # Full simulated descent trajectory — orange (matches descent viewer trail)
    pl.add_mesh(pv.Spline(descent_xyz, len(descent_xyz)),
                color=[1.0, 0.55, 0.0], line_width=3,
                label='Descent (simulated)')

    # Markers
    mk = R_moon * 0.007
    pl.add_mesh(pv.Sphere(radius=mk,       center=orbit_xyz[-1]),
                color=[1.0, 0.55, 0.0], label='Deorbit burn')
    pl.add_mesh(pv.Sphere(radius=mk * 0.5, center=descent_xyz[0]),
                color=[0.85, 0.95, 1.0], label='Descent start')
    pl.add_mesh(pv.Sphere(radius=mk * 0.5, center=descent_xyz[-1]),
                color=[0.0, 0.85, 0.85], label='Touchdown')

    pl.add_legend(loc='lower right', size=(0.28, 0.22),
                  bcolor=[0.0, 0.0, 0.0, 0.5])

    orbit_r = float(np.linalg.norm(orbit_xyz[0]))
    pl.camera.position    = (orbit_r * 2.2, orbit_r * 1.0, orbit_r * 1.6)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up          = (0.0, 0.0, 1.0)

    pl.show()
