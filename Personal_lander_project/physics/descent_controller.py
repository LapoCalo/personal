"""
Feedback controller for powered descent.

Strategy
--------
Two-phase controller producing a smooth curved descent arc:

  Phase 1 (Braking phase, alt > 2000m):
    Decelerate in ALL directions (radial + horizontal) simultaneously.
    Thrust is applied proportionally to the velocity vector:
        a_des = -alpha * [w, u, v]   (decel toward zero velocity)
        T = clip(mass * a_des, 0, T_max)
    This creates a smooth, realistic gravity-turn style arc where the
    lander's trajectory curves from near-horizontal (at orbit) to more
    vertical as it descends.

  Phase 2 (Landing phase, alt <= 2000m):
    Focus on the vertical (radial) component only.
    Use parabolic braking reference:
        w_ref = -k * sqrt(2 * g * alt)
    with PD control:
        a_cmd = -(g + Kp * e_alt + Kd * (w - w_ref))
    This ensures a soft touchdown with near-zero vertical velocity.

Telemetry
---------
Records internal signals AND the 3D thrust vector components for:
  - Post-flight analysis
  - Real-time lander attitude control in the 3D viewer
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class TelemetryLog:
    """Stores controller internal signals at each integrator call."""
    times: List[float]       = field(default_factory=list)
    alt_errors: List[float]  = field(default_factory=list)
    vel_errors: List[float]  = field(default_factory=list)
    accel_cmds: List[float]  = field(default_factory=list)
    thrust_cmds: List[float] = field(default_factory=list)
    thrust_r: List[float]    = field(default_factory=list)   # radial component
    thrust_e: List[float]    = field(default_factory=list)   # east component
    thrust_n: List[float]    = field(default_factory=list)   # north component

    def record(self, t, e_alt, e_vel, a_cmd, T_cmd, T_r=0.0, T_e=0.0, T_n=0.0):
        self.times.append(t)
        self.alt_errors.append(e_alt)
        self.vel_errors.append(e_vel)
        self.accel_cmds.append(a_cmd)
        self.thrust_cmds.append(T_cmd)
        self.thrust_r.append(T_r)
        self.thrust_e.append(T_e)
        self.thrust_n.append(T_n)

    def as_arrays(self):
        import numpy as np
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


class PDDescentController:
    """
    Two-phase descent controller: smooth braking arc → soft landing.

    Parameters
    ----------
    moon_params : dict
        Needs 'grav_param_m3s2' and 'radius_m'.
    vehicle_params : dict
        Needs 'thrust_max_N', 'thrust_min_N'.
    omega_n : float
        Closed-loop natural frequency [rad/s] for the landing phase.
    zeta : float
        Damping ratio (>1 = overdamped).
    kd_scale : float
        Extra multiplier on derivative gain.
    brake_gain : float
        Gain k in the parabolic braking law.
    target_altitude_m : float
        Desired final altitude [m].
    """

    def __init__(
        self,
        moon_params,
        vehicle_params,
        omega_n=0.25,
        zeta=1.2,
        kd_scale=35.0,
        Ki=0.005,
        brake_gain=0.4,
        target_altitude_m=1.0,
    ):
        self.GM         = float(moon_params['grav_param_m3s2'])
        self.R_moon     = float(moon_params['radius_m'])
        self.T_max      = float(vehicle_params['thrust_max_N'])
        self.T_min      = float(vehicle_params['thrust_min_N'])

        self.Kp         = omega_n ** 2
        self.Kd         = kd_scale * zeta * omega_n
        self.Ki         = Ki
        self.brake_gain = brake_gain
        self.h_target   = target_altitude_m

        # Phase threshold
        self.h_landing   = 2000.0   # [m] — switch to PID at this altitude
        self.alpha_brake = 0.0085   # [1/s] — decel rate during braking phase

        # Integral state (reset when entering landing phase)
        self._integral   = 0.0
        self._t_prev     = None
        # Anti-windup: limit integral contribution to ±g_local
        self._i_max      = 1.62 / max(Ki, 1e-9)   # [m·s]

        self.log = TelemetryLog()

    def _local_gravity(self, r):
        return self.GM / r**2

    def _reference_descent_rate(self, alt_error, g_local):
        """Parabolic braking law: w_ref = -k * sqrt(2 * g * Δh)."""
        return -self.brake_gain * np.sqrt(2.0 * g_local * max(alt_error, 0.0))

    def compute_thrust(self, t, state):
        """
        Return thrust vector [T_radial, T_east, T_north] in Newtons.

        Parameters
        ----------
        t     : float
        state : array-like  [r, lat, lon, w, u, v, mass]
        """
        r, lat, lon, w, u, v, mass = (float(x) for x in state)

        g_local = self._local_gravity(r)
        alt     = r - self.R_moon

        # ------------------------------------------------------------------
        # Phase selection
        # ------------------------------------------------------------------
        if alt > self.h_landing:
            # --- Phase 1: Braking (smooth curve descent) ---
            # Decelerate in all directions proportionally to velocity
            v_vec = np.array([w, u, v], dtype=float)
            a_des = -self.alpha_brake * v_vec

            T_des = mass * a_des
            T_mag = float(np.linalg.norm(T_des))

            if T_mag > self.T_max:
                T_vec = self.T_max * T_des / T_mag
            else:
                T_vec = T_des

            T_r, T_e, T_n = float(T_vec[0]), float(T_vec[1]), float(T_vec[2])
            a_cmd = self.alpha_brake * np.linalg.norm(v_vec)
            e_alt_log = alt - self.h_target
            e_vel_log = 0.0

        elif alt > 100.0:
            # --- Phase 2: Landing (vertical PID) ---
            e_alt = alt - self.h_target
            w_ref = self._reference_descent_rate(e_alt, g_local)
            e_vel = w - w_ref

            # Integral: accumulate altitude error, with anti-windup clamp
            dt = (t - self._t_prev) if self._t_prev is not None else 0.0
            self._integral = float(np.clip(
                self._integral + e_alt * dt,
                -self._i_max, self._i_max
            ))

            a_cmd = -(g_local + self.Kp * e_alt + self.Ki * self._integral + self.Kd * e_vel)
            T_r = float(np.clip(mass * a_cmd, self.T_min, self.T_max))
            T_e = T_n = 0.0
            e_alt_log = e_alt
            e_vel_log = e_vel

        else:
            # --- Phase 3: Final approach (alt <= 100m) — PID vertical + horiz damping ---
            # Always damp horizontal velocity proportionally — no threshold.
            # This drives v_horiz to zero smoothly without leaving residual.
            v_horiz = float(np.sqrt(u**2 + v**2))

            if v_horiz > 0.01:
                # Proportional horizontal damping: acceleration = Kd * v_horiz,
                # capped at 60% of T_max so vertical still gets plenty of budget.
                Kd_horiz = 3.0   # [1/s] — strong enough to kill v_horiz in <1 s
                a_horiz   = min(Kd_horiz * v_horiz, self.T_max / mass * 0.6)
                T_horiz   = mass * a_horiz
                T_e = -T_horiz * (u / v_horiz)
                T_n = -T_horiz * (v / v_horiz)
                T_vert_budget = float(np.sqrt(max(0.0, self.T_max**2 - T_horiz**2)))
            else:
                T_e = T_n = 0.0
                T_vert_budget = self.T_max

            # Vertical PID with available budget
            e_alt = alt - self.h_target
            w_ref = self._reference_descent_rate(e_alt, g_local)
            e_vel = w - w_ref

            dt = (t - self._t_prev) if self._t_prev is not None else 0.0
            self._integral = float(np.clip(
                self._integral + e_alt * dt,
                -self._i_max, self._i_max
            ))

            a_cmd = -(g_local + self.Kp * e_alt + self.Ki * self._integral + self.Kd * e_vel)
            T_r = float(np.clip(mass * a_cmd, self.T_min, T_vert_budget))
            e_alt_log = e_alt
            e_vel_log = e_vel

        # Reset integral when transitioning into landing phase
        if self._t_prev is not None:
            was_braking = (r - self.R_moon + w * (t - self._t_prev)) > self.h_landing
            if was_braking and alt <= self.h_landing:
                self._integral = 0.0

        self._t_prev = t

        T_cmd_mag = float(np.sqrt(T_r**2 + T_e**2 + T_n**2))
        self.log.record(t, e_alt_log, e_vel_log, a_cmd, T_cmd_mag, T_r, T_e, T_n)
        return np.array([T_r, T_e, T_n])
