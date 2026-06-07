\# Lunar Powered Descent Simulator



A physics-based simulation of a lunar powered descent, built from scratch in Python. The spacecraft departs from a 100 km circular parking orbit around the Moon, performs a deorbit burn, and executes a fully autonomous landing using a three-phase feedback controller. The project includes 2D diagnostic plots, a full-mission 3D globe viewer, and a close-up animated terminal descent viewer with optional GIF export.



\---



\## Features



\- \*\*Full-mission simulation\*\* from circular parking orbit to touchdown, including deorbit burn and powered descent

\- \*\*Rotating Moon-fixed equations of motion\*\* in spherical coordinates with Coriolis, centrifugal, and curvature terms

\- \*\*Three-phase autonomous controller\*\*: braking arc → vertical PID → final approach with horizontal damping

\- \*\*3D globe viewer\*\* showing the parking orbit, deorbit point, and full descent arc on a textured Moon sphere

\- \*\*Close-up terminal descent animation\*\* with a 3D lander model, live telemetry overlay, and dynamic thrust-based attitude

\- \*\*GIF export\*\* of the terminal descent animation

\- \*\*2D diagnostic plots\*\*: altitude, velocity components, propellant consumption, and controller telemetry



\---



\## Project Structure



```

.

├── mission.py                  # Entry point — run this

├── config/

│   ├── moon.yaml               # Lunar physical constants

│   ├── lander.yaml             # Vehicle parameters (mass, thrust, Isp)

│   └── sim.yaml                # Integrator settings and initial conditions

├── physics/

│   ├── lunar\_eom.py            # Equations of motion (rotating spherical frame)

│   ├── descent\_controller.py   # Three-phase PID/PD controller + telemetry log

│   └── orbital\_mechanics.py    # Parking orbit and descent arc generator

├── simulation/

│   ├── integrator.py           # RK45 wrapper, touchdown event detection

│   ├── mission\_overview.py     # Full-mission 3D globe viewer (PyVista)

│   ├── descent\_viewer.py       # Terminal descent 3D animation + GIF export

│   └── mission\_plots.py        # 2D Matplotlib diagnostic plots

└── assets/

&#x20;   └── moon\_texture.jpg        # Auto-downloaded on first run (\~3 MB, NASA)

```



\---



\## Installation



Python 3.9+ is recommended.



```bash

pip install numpy scipy pandas matplotlib pyvista pyyaml Pillow

```



\---



\## Usage



Run the full mission from the project root:



```bash

python mission.py

```



This will:

1\. Compute a 100 km circular parking orbit

2\. Simulate the powered descent from deorbit burn to touchdown

3\. Display 2D diagnostic plots (close to continue)

4\. Open the full-mission 3D globe viewer (close to continue)

5\. Run the close-up terminal descent animation and save `descent.gif`



On the first run, the Moon texture (\~3 MB) is automatically downloaded from NASA and cached to `assets/moon\_texture.jpg`.



\---



\## Configuration



All parameters are in the `config/` directory and can be edited without touching the code.



\*\*`config/lander.yaml`\*\* — Vehicle parameters



| Parameter | Default | Description |

|---|---|---|

| `dry\_mass\_kg` | 500 kg | Structural mass |

| `propellant\_mass\_kg` | 700 kg | Initial propellant load |

| `thrust\_max\_N` | 15 000 N | Maximum throttleable thrust |

| `specific\_impulse\_s` | 311 s | Engine Isp |



\*\*`config/moon.yaml`\*\* — Lunar constants (NASA lunar fact sheet)



| Parameter | Value | Description |

|---|---|---|

| `radius\_m` | 1 737 400 m | Mean lunar radius |

| `grav\_param\_m3s2` | 4.9028 × 10¹² m³/s² | Gravitational parameter GM |

| `spin\_rate\_rads` | 2.6617 × 10⁻⁶ rad/s | Sidereal rotation rate |



\*\*`config/sim.yaml`\*\* — Integrator and initial conditions



| Parameter | Default | Description |

|---|---|---|

| `method` | RK45 | SciPy integrator method |

| `dt\_output\_s` | 0.02 s | Output sample interval (50 Hz) |

| `max\_step\_s` | 0.5 s | Hard cap on integrator step size |

| `max\_vertical\_speed\_ms` | 2.0 m/s | Safe landing vertical speed limit |



\---



\## Physics



The equations of motion are integrated in a \*\*rotating Moon-fixed spherical frame\*\* with state vector `\[r, lat, lon, v\_r, v\_e, v\_n, mass]`. Thrust is a full 3D vector `\[T\_r, T\_e, T\_n]` applied independently to the radial, eastward, and northward channels. Mass depletion follows the Tsiolkovsky rocket equation at each timestep.



The non-inertial acceleration terms include:

\- Central gravity field (μ/r²)

\- Coriolis acceleration (2Ω × v)

\- Centrifugal acceleration (Ω² × r)

\- Spherical geometry curvature terms



\---



\## Controller



Landing is managed by a \*\*three-phase feedback controller\*\*:



| Phase | Altitude | Strategy |

|---|---|---|

| \*\*1 — Braking\*\* | > 2 000 m | Proportional deceleration in all three axes simultaneously, producing a smooth gravity-turn-style arc |

| \*\*2 — Vertical PID\*\* | 100 – 2 000 m | PID on vertical speed with parabolic braking reference `w\_ref = -k √(2g·Δh)` and anti-windup integral clamp |

| \*\*3 — Final approach\*\* | < 100 m | Vertical PID with added proportional horizontal damping to null residual lateral drift |



Touchdown targets: vertical speed < 2 m/s, horizontal speed < 0.1 m/s.



\---



\## GIF Export



The terminal descent animation is automatically saved to `descent.gif` in the project root at the end of each run. Each run overwrites the previous file.



To disable GIF export, set `save\_gif=False` in the `\_launch\_descent` call inside `mission.py`.



\---



\## References



\- Battin, R. H. — \*An Introduction to the Mathematics and Methods of Astrodynamics\*, AIAA, 1999

\- Wie, B. — \*Space Vehicle Dynamics and Control\*, AIAA, 2008

\- NASA Lunar Fact Sheet — https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html

\- Moon texture — NASA Visible Earth / LROC, public domain

