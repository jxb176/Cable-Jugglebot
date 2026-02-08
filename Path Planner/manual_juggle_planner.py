#!/usr/bin/env python3
"""
manual_juggle_planner_demo_jugglepath.py

Manual juggling demo that:
  1) Uses event-based CATCH/THROW definitions (like manual_juggle_planner)
  2) Generates the *hand* trajectory using jugglepath.py (JugglePath + time laws)
  3) Simulates the *ball* as HELD (follows hand) or FLIGHT (ballistic)

This is meant as a "bridge demo" to move manual_juggle_planner onto JugglePath.

Notes:
- JugglePath's s_curve / s_curve_monotonic produce min-time profiles, not fixed-duration.
  To match event timing, we time-stretch each generated segment to the desired dt.
- jugglepath.py does not include a Wait primitive; we implement a simple sampled wait here.

Keys in animation:
  space : play/pause
  left  : step backward
  right : step forward
  r     : restart
  esc   : close
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from jugglepath import State3D, JugglePath


# ----------------------------
# Plotting helper
# ----------------------------
def place_figure(fig, x: int, y: int, w: int, h: int):
    """Move/resize a matplotlib figure window in screen pixels (Qt/Tk backends)."""
    mgr = fig.canvas.manager
    try:
        mgr.window.setGeometry(x, y, w, h)  # Qt
    except Exception:
        try:
            mgr.window.wm_geometry(f"{w}x{h}+{x}+{y}")  # Tk
        except Exception:
            pass


# ----------------------------
# Core math
# ----------------------------
def ballistic_v0(p_release: np.ndarray, p_catch: np.ndarray, T: float, g: np.ndarray) -> np.ndarray:
    """Solve v0 for ballistic flight."""
    if T <= 0:
        raise ValueError("Throw duration T must be > 0")
    return (p_catch - p_release - 0.5 * g * (T**2)) / T


def ballistic_state(p0: np.ndarray, v0: np.ndarray, g: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (p(t), v(t)) for ballistic motion under constant gravity."""
    p = p0 + v0 * t + 0.5 * g * (t**2)
    v = v0 + g * t
    return p, v


# ----------------------------
# Workspace (optional, keep for later constraints)
# ----------------------------
@dataclass
class WorkspaceCylinder:
    radius: float
    z_min: float
    z_max: float


# ----------------------------
# Events / Ball segments
# ----------------------------
@dataclass
class Event:
    t: float
    kind: str        # "CATCH" or "THROW"
    ball_id: int
    p: np.ndarray    # (3,)
    # For THROW only:
    duration: Optional[float] = None
    p_catch: Optional[np.ndarray] = None
    v_hand: Optional[np.ndarray] = None   # desired hand velocity at the event (3,)
    vel_scale: float = 1.0


@dataclass
class Segment:
    t0: float
    t1: float
    kind: str  # "HELD" or "FLIGHT"

    # HELD: follows hand
    # FLIGHT: ballistic initial conditions
    p_release: Optional[np.ndarray] = None
    v_release: Optional[np.ndarray] = None


class Ball:
    def __init__(self, ball_id: int, radius: float = 0.03):
        self.ball_id = int(ball_id)
        self.radius = float(radius)
        self.segments: List[Segment] = []

    def build_from_events(self, events: List[Event], g: np.ndarray):
        """Build HELD/FLIGHT segments from CATCH/THROW."""
        ev = [e for e in events if e.ball_id == self.ball_id]
        ev = sorted(ev, key=lambda e: e.t)
        if not ev:
            raise ValueError(f"Ball {self.ball_id}: no events")

        self.segments = []
        i = 0
        while i < len(ev) - 1:
            a = ev[i]
            b = ev[i + 1]
            ka = a.kind.upper()
            kb = b.kind.upper()

            if b.t <= a.t:
                raise ValueError(f"Ball {self.ball_id}: non-increasing times at {a.t} -> {b.t}")

            if ka == "CATCH" and kb == "THROW":
                # HELD interval is implicit: ball follows hand
                self.segments.append(Segment(t0=a.t, t1=b.t, kind="HELD"))
                i += 1
                continue

            if ka == "THROW":
                if a.duration is None or a.p_catch is None:
                    raise ValueError(f"Ball {self.ball_id}: THROW at t={a.t} missing duration or p_catch")

                t_end = a.t + a.duration
                eps = 1e-9
                if not (kb == "CATCH" and abs(b.t - t_end) <= eps):
                    raise ValueError(
                        f"Ball {self.ball_id}: THROW at t={a.t} expects CATCH at t={t_end}, "
                        f"but next event is {b.kind} at t={b.t}"
                    )

                v0 = ballistic_v0(a.p, a.p_catch, a.duration, g)
                self.segments.append(Segment(
                    t0=a.t, t1=t_end, kind="FLIGHT",
                    p_release=a.p.copy(),
                    v_release=v0.copy(),
                ))
                i += 1
                continue

            # fallback: treat as HELD
            self.segments.append(Segment(t0=a.t, t1=b.t, kind="HELD"))
            i += 1

    def state_at(self, t: float, g: np.ndarray, hand_eval) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (p,v,a,j) at time t."""
        if not self.segments:
            z = np.zeros(3)
            return z, z, z, z

        # clamp outside time range
        if t <= self.segments[0].t0:
            s0 = self.segments[0]
            if s0.kind == "HELD":
                return hand_eval(t)
            p = s0.p_release.copy()
            v = s0.v_release.copy()
            return p, v, g.copy(), np.zeros(3)

        if t >= self.segments[-1].t1:
            sl = self.segments[-1]
            if sl.kind == "HELD":
                return hand_eval(t)
            p_end, v_end = ballistic_state(sl.p_release, sl.v_release, g, sl.t1 - sl.t0)
            return p_end, v_end, g.copy(), np.zeros(3)

        for s in self.segments:
            if s.t0 <= t <= s.t1:
                if s.kind == "HELD":
                    return hand_eval(t)
                dt = t - s.t0
                p, v = ballistic_state(s.p_release, s.v_release, g, dt)
                return p, v, g.copy(), np.zeros(3)

        z = np.zeros(3)
        return z, z, z, z


# ----------------------------
# Hand trajectory via JugglePath
# ----------------------------
@dataclass
class HandWaypoint:
    t: float
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray


@dataclass
class HandSegSampled:
    t0: float
    t1: float
    T: np.ndarray         # (N,)
    P: np.ndarray         # (N,3)
    V: np.ndarray         # (N,3)
    A: np.ndarray         # (N,3)
    J: np.ndarray         # (N,3)


class HandTrajectoryJugglePath:
    """
    Build a piecewise trajectory that matches waypoint timing exactly, but uses JugglePath
    to generate the shape (then time-stretch per segment as needed).

    Per segment:
      - Build a 2-waypoint JugglePath
      - Use time_law in {"linear","s_curve","s_curve_monotonic"}
      - Stretch to match (t1 - t0)
      - Resample onto uniform dt grid
    """

    def __init__(
        self,
        ws: WorkspaceCylinder,
        accel_ref: float = 50.0,
        jerk_ref: float = 2000.0,
        default_time_law: str = "s_curve",  # good default for hand motion
    ):
        self.ws = ws
        self.accel_ref = float(accel_ref)
        self.jerk_ref = float(jerk_ref)
        self.default_time_law = str(default_time_law)

        self.segs: List[HandSegSampled] = []
        self._last_p = np.zeros(3)

    @staticmethod
    def _time_stretch_traj(traj: np.ndarray, scale: float) -> np.ndarray:
        """
        Stretch duration by 'scale' while preserving geometry.

          t' = t * scale
          v' = v / scale
          a' = a / scale^2
          j' = j / scale^3
        """
        scale = float(scale)
        if abs(scale - 1.0) < 1e-12:
            return traj
        out = traj.copy()
        out[:, 0] *= scale
        out[:, 4:7] /= scale
        out[:, 7:10] /= (scale * scale)
        out[:, 10:13] /= (scale * scale * scale)
        return out

    @staticmethod
    def _sample_wait(p: np.ndarray, t0: float, t1: float, dt: float) -> HandSegSampled:
        """Sample a constant pose wait segment."""
        if t1 <= t0:
            T = np.array([t0], dtype=float)
        else:
            n = max(2, int(np.ceil((t1 - t0) / dt)) + 1)
            T = np.linspace(t0, t1, n)
        P = np.tile(p.reshape(1, 3), (len(T), 1))
        Z = np.zeros((len(T), 3), dtype=float)
        return HandSegSampled(t0=t0, t1=t1, T=T, P=P, V=Z.copy(), A=Z.copy(), J=Z.copy())

    def build_from_waypoints(
        self,
        wps: List[HandWaypoint],
        dt: float = 0.005,
        time_laws: Optional[List[str]] = None,
    ):
        """
        Build from explicit waypoints.
        - time_laws: optional list of per-segment time laws, length N-1
        """
        wps = sorted(wps, key=lambda w: float(w.t))
        if len(wps) < 1:
            self.segs = []
            return

        # Ensure strictly increasing times
        eps = 1e-9
        for i in range(len(wps) - 1):
            if wps[i + 1].t <= wps[i].t + eps:
                raise ValueError(f"Hand waypoints must be increasing in time: {wps[i].t} -> {wps[i+1].t}")

        if time_laws is None:
            time_laws = [self.default_time_law] * (len(wps) - 1)
        if len(time_laws) != (len(wps) - 1):
            raise ValueError("time_laws must be length N-1 if provided")

        self.segs = []
        self._last_p = wps[0].p.copy()

        for i, (a_wp, b_wp) in enumerate(zip(wps[:-1], wps[1:])):
            t0 = float(a_wp.t)
            t1 = float(b_wp.t)
            dt_avail = t1 - t0
            if dt_avail <= 0:
                continue

            p0 = a_wp.p.copy()
            p1 = b_wp.p.copy()

            # Same position -> wait
            if np.linalg.norm(p1 - p0) < 1e-12:
                seg = self._sample_wait(p0, t0, t1, dt)
                self.segs.append(seg)
                self._last_p = p1.copy()
                continue

            # Build a tiny JugglePath for this segment
            start = State3D(p=p0, v=a_wp.v.copy(), a=a_wp.a.copy())
            jp = JugglePath(sample_hz=1.0 / dt, start=start)
            jp.add_segment(
                p=p1,
                v=b_wp.v.copy(),
                a=b_wp.a.copy(),
                t=dt_avail,  # used only for linear; still OK to set
                time_law=time_laws[i],
                accel_ref=self.accel_ref,
                jerk_ref=self.jerk_ref,
            )
            res = jp.build()
            traj = res.traj.copy()

            # Time-stretch to hit desired dt_avail
            T_nom = float(traj[-1, 0]) if traj.shape[0] else 0.0
            if T_nom <= 1e-12:
                seg = self._sample_wait(p0, t0, t1, dt)
                self.segs.append(seg)
                self._last_p = p1.copy()
                continue

            scale = dt_avail / T_nom
            traj = self._time_stretch_traj(traj, scale=scale)

            # Offset to absolute time
            traj[:, 0] += t0

            # Resample onto a uniform grid in [t0, t1]
            n = max(2, int(np.ceil((t1 - t0) / dt)) + 1)
            T = np.linspace(t0, t1, n)

            P = np.column_stack([np.interp(T, traj[:, 0], traj[:, 1]),
                                 np.interp(T, traj[:, 0], traj[:, 2]),
                                 np.interp(T, traj[:, 0], traj[:, 3])])
            V = np.column_stack([np.interp(T, traj[:, 0], traj[:, 4]),
                                 np.interp(T, traj[:, 0], traj[:, 5]),
                                 np.interp(T, traj[:, 0], traj[:, 6])])
            A = np.column_stack([np.interp(T, traj[:, 0], traj[:, 7]),
                                 np.interp(T, traj[:, 0], traj[:, 8]),
                                 np.interp(T, traj[:, 0], traj[:, 9])])
            J = np.column_stack([np.interp(T, traj[:, 0], traj[:, 10]),
                                 np.interp(T, traj[:, 0], traj[:, 11]),
                                 np.interp(T, traj[:, 0], traj[:, 12])])

            self.segs.append(HandSegSampled(t0=t0, t1=t1, T=T, P=P, V=V, A=A, J=J))
            self._last_p = p1.copy()

    def eval_at(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.segs:
            z = np.zeros(3)
            return z, z, z, z

        for s in self.segs:
            if s.t0 <= t <= s.t1:
                i = int(np.clip(np.searchsorted(s.T, t), 0, len(s.T) - 1))
                return s.P[i].copy(), s.V[i].copy(), s.A[i].copy(), s.J[i].copy()

        # Outside: hold last pose
        p = self._last_p.copy()
        z = np.zeros(3)
        return p, z, z, z


# ----------------------------
# Planner
# ----------------------------
class ManualJugglingPlanner:
    def __init__(self, g_mag: float = 9.81, ws: Optional[WorkspaceCylinder] = None):
        self.g = np.array([0.0, 0.0, -abs(float(g_mag))], dtype=float)
        self.ws = ws if ws is not None else WorkspaceCylinder(radius=0.25, z_min=-1.0, z_max=1.0)

        self.events: List[Event] = []
        self.balls: Dict[int, Ball] = {}

        self.hand = HandTrajectoryJugglePath(ws=self.ws, accel_ref=50.0, jerk_ref=2000.0, default_time_law="s_curve")

    def get_ball(self, ball_id: int) -> Ball:
        if ball_id not in self.balls:
            self.balls[ball_id] = Ball(ball_id)
        return self.balls[ball_id]

    def add_catch(self, t: float, ball_id: int, p_xyz: Tuple[float, float, float], v_hand=(0.0, 0.0, 0.0)):
        self.get_ball(ball_id)
        self.events.append(Event(
            t=float(t),
            kind="CATCH",
            ball_id=int(ball_id),
            p=np.array(p_xyz, dtype=float),
            v_hand=np.array(v_hand, dtype=float),
        ))

    def add_throw(
        self,
        t: float,
        ball_id: int,
        p_release: Tuple[float, float, float],
        duration: float,
        p_catch: Tuple[float, float, float],
        v_hand=(0.0, 0.0, 0.0),
    ):
        self.get_ball(ball_id)
        self.events.append(Event(
            t=float(t),
            kind="THROW",
            ball_id=int(ball_id),
            p=np.array(p_release, dtype=float),
            duration=float(duration),
            p_catch=np.array(p_catch, dtype=float),
            v_hand=np.array(v_hand, dtype=float),
        ))

    def _build_hand_waypoints_demo(self) -> List[HandWaypoint]:
        """
        Build a hand waypoint list from events PLUS a pre-throw stroke waypoint,
        mimicking testprofile_single_throw_static_catch:
          start at (0,0,0)
          dip to z=-0.2
          return to z=0 with vz=+6 at throw
          (then hand remains at catch point afterwards)
        """
        ev = sorted([e for e in self.events if e.kind.upper() in ("CATCH", "THROW")], key=lambda e: e.t)
        if not ev:
            return []

        # We'll assume single ball demo and use the first catch/throw as anchor.
        # Start waypoint at first event
        wps: List[HandWaypoint] = []
        e0 = ev[0]
        wps.append(HandWaypoint(t=e0.t, p=e0.p.copy(), v=(e0.v_hand.copy() if e0.v_hand is not None else np.zeros(3)), a=np.zeros(3)))

        # Insert one stroke waypoint before the first THROW if possible
        # Find first THROW
        throws = [e for e in ev if e.kind.upper() == "THROW"]
        if throws:
            thr = throws[0]
            t_throw = float(thr.t)
            # Put dip at 0.2s after start if there's room; otherwise at midpoint
            t_start = float(e0.t)
            t_dip = min(t_throw - 0.05, t_start + 0.20)
            if t_dip > t_start + 1e-6:
                p_dip = thr.p.copy()
                p_dip[2] = -0.2
                wps.append(HandWaypoint(t=t_dip, p=p_dip, v=np.zeros(3), a=np.zeros(3)))

        # Add event waypoints (catch/throw), preserving their desired velocities
        for e in ev[1:]:
            v = e.v_hand.copy() if e.v_hand is not None else np.zeros(3)
            wps.append(HandWaypoint(t=float(e.t), p=e.p.copy(), v=v, a=np.zeros(3)))

        # If there is a catch after a throw, keep holding there for a short time (optional)
        t_last = wps[-1].t
        wps.append(HandWaypoint(t=t_last + 0.30, p=wps[-1].p.copy(), v=np.zeros(3), a=np.zeros(3)))

        # Ensure strictly increasing & sorted
        wps = sorted(wps, key=lambda w: float(w.t))
        # Drop accidental duplicates in time
        cleaned = [wps[0]]
        for w in wps[1:]:
            if w.t > cleaned[-1].t + 1e-9:
                cleaned.append(w)
        return cleaned

    def build(self, dt: float = 0.005):
        # Build hand trajectory (using demo waypoint builder)
        wps = self._build_hand_waypoints_demo()
        if not wps:
            raise ValueError("No hand waypoints (need at least one CATCH/THROW event).")

        # Segment-wise time laws:
        # - Dip segment: s_curve (gentle)
        # - Up to throw: s_curve_monotonic (good for dv shaping)
        # - After throw/catch: s_curve
        time_laws = []
        for a, b in zip(wps[:-1], wps[1:]):
            # if target has positive vz, prefer monotonic dv primitive
            if b.v[2] > 1e-6:
                time_laws.append("s_curve_monotonic")
            else:
                time_laws.append("s_curve")

        self.hand.build_from_waypoints(wps, dt=dt, time_laws=time_laws)

        # Build ball segments
        for ball in self.balls.values():
            ball.build_from_events(self.events, self.g)

    def simulate(self, dt: float = 0.005) -> Dict[str, np.ndarray]:
        self.build(dt=dt)

        # time span from events/hand segs
        t0 = min(e.t for e in self.events)
        t1 = max(e.t + (e.duration or 0.0) for e in self.events)
        t1 = max(t1, max(seg.t1 for seg in self.hand.segs))

        n = max(2, int(np.ceil((t1 - t0) / dt)) + 1)
        T = np.linspace(t0, t1, n)

        HP = np.zeros((n, 3)); HV = np.zeros((n, 3)); HA = np.zeros((n, 3)); HJ = np.zeros((n, 3))
        BP = np.zeros((n, 3)); BV = np.zeros((n, 3)); BA = np.zeros((n, 3)); BJ = np.zeros((n, 3))

        def hand_eval(tt: float):
            return self.hand.eval_at(tt)

        # single-ball demo: pick lowest id
        bid = sorted(self.balls.keys())[0]
        ball = self.balls[bid]

        for i, tt in enumerate(T):
            hp, hv, ha, hj = hand_eval(float(tt))
            bp, bv, ba, bj = ball.state_at(float(tt), self.g, hand_eval)

            HP[i], HV[i], HA[i], HJ[i] = hp, hv, ha, hj
            BP[i], BV[i], BA[i], BJ[i] = bp, bv, ba, bj

        return {"T": T, "HP": HP, "HV": HV, "HA": HA, "HJ": HJ, "BP": BP, "BV": BV, "BA": BA, "BJ": BJ}

    def plot_timeseries(self, sim: Dict[str, np.ndarray]):
        t = sim["T"]
        HP, HV, HA, HJ = sim["HP"], sim["HV"], sim["HA"], sim["HJ"]
        BP, BV, BA, BJ = sim["BP"], sim["BV"], sim["BA"], sim["BJ"]

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(11, 9))
        axs[0].set_title("Hand (solid) vs Ball (dashed) kinematics")

        # Position
        axs[0].plot(t, HP[:, 0], label="Hx"); axs[0].plot(t, HP[:, 1], label="Hy"); axs[0].plot(t, HP[:, 2], label="Hz")
        axs[0].plot(t, BP[:, 0], linestyle="--"); axs[0].plot(t, BP[:, 1], linestyle="--"); axs[0].plot(t, BP[:, 2], linestyle="--")
        axs[0].set_ylabel("pos [m]")
        axs[0].legend(loc="upper right", ncol=3)

        # Velocity
        axs[1].plot(t, HV[:, 0]); axs[1].plot(t, HV[:, 1]); axs[1].plot(t, HV[:, 2])
        axs[1].plot(t, BV[:, 0], linestyle="--"); axs[1].plot(t, BV[:, 1], linestyle="--"); axs[1].plot(t, BV[:, 2], linestyle="--")
        axs[1].set_ylabel("vel [m/s]")

        # Acceleration
        axs[2].plot(t, HA[:, 0]); axs[2].plot(t, HA[:, 1]); axs[2].plot(t, HA[:, 2])
        axs[2].plot(t, BA[:, 0], linestyle="--"); axs[2].plot(t, BA[:, 1], linestyle="--"); axs[2].plot(t, BA[:, 2], linestyle="--")
        axs[2].set_ylabel("acc [m/s²]")

        # Jerk
        axs[3].plot(t, HJ[:, 0]); axs[3].plot(t, HJ[:, 1]); axs[3].plot(t, HJ[:, 2])
        axs[3].plot(t, BJ[:, 0], linestyle="--"); axs[3].plot(t, BJ[:, 1], linestyle="--"); axs[3].plot(t, BJ[:, 2], linestyle="--")
        axs[3].set_ylabel("jerk [m/s³]")
        axs[3].set_xlabel("time [s]")

        fig.tight_layout()
        return fig

    def animate(self, sim: Dict[str, np.ndarray], stride: int = 2, trail: int = 200):
        T = sim["T"][::stride]
        HP = sim["HP"][::stride]
        BP = sim["BP"][::stride]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Hand (blue) + Ball (orange)")

        # Bounds
        P_all = np.vstack([HP, BP])
        xmin, ymin, zmin = np.min(P_all, axis=0)
        xmax, ymax, zmax = np.max(P_all, axis=0)
        pad = 0.05
        xmin -= pad; xmax += pad
        ymin -= pad; ymax += pad
        zmin -= pad; zmax += pad
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        half = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)

        hand_marker = ax.plot([], [], [], marker="o", linestyle="None")[0]
        ball_marker = ax.plot([], [], [], marker="o", linestyle="None")[0]
        hand_trail = ax.plot([], [], [], linewidth=1.5, alpha=0.8)[0]
        ball_trail = ax.plot([], [], [], linewidth=1.5, alpha=0.8)[0]
        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

        state = {"paused": False, "i": 0}

        def set_artists(i: int):
            time_text.set_text(f"t = {float(T[i]):.3f} s")

            ph = HP[i]
            pb = BP[i]

            hand_marker.set_data([ph[0]], [ph[1]])
            hand_marker.set_3d_properties([ph[2]])

            ball_marker.set_data([pb[0]], [pb[1]])
            ball_marker.set_3d_properties([pb[2]])

            k0 = max(0, i - int(trail))
            trh = HP[k0:i + 1]
            trb = BP[k0:i + 1]

            hand_trail.set_data(trh[:, 0], trh[:, 1])
            hand_trail.set_3d_properties(trh[:, 2])

            ball_trail.set_data(trb[:, 0], trb[:, 1])
            ball_trail.set_3d_properties(trb[:, 2])

            return [time_text, hand_marker, ball_marker, hand_trail, ball_trail]

        def on_key(event):
            if event.key == " ":
                state["paused"] = not state["paused"]
            elif event.key == "right":
                state["i"] = min(state["i"] + 1, len(T) - 1)
                set_artists(state["i"]); fig.canvas.draw_idle()
            elif event.key == "left":
                state["i"] = max(state["i"] - 1, 0)
                set_artists(state["i"]); fig.canvas.draw_idle()
            elif event.key == "r":
                state["i"] = 0
            elif event.key == "escape":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

        def update(_frame):
            if not state["paused"]:
                state["i"] = min(state["i"] + 1, len(T) - 1)
            return set_artists(state["i"])

        ani = FuncAnimation(fig, update, interval=20, blit=False)
        return fig, ani


# ----------------------------
# Demo main
# ----------------------------
def main():
    planner = ManualJugglingPlanner(
        g_mag=9.81,
        ws=WorkspaceCylinder(radius=0.25, z_min=-1.0, z_max=1.0),
    )

    # --- Single-ball self throw/catch at origin ---
    planner.get_ball(0)

    # Catch at origin (start held)
    planner.add_catch(t=0.00, ball_id=0, p_xyz=(0.0, 0.0, 0.0), v_hand=(0.0, 0.0, 0.0))

    # Throw at t=0.70 from origin with upward hand velocity (this becomes the "throw waypoint velocity")
    # Catch after 1.00s at origin
    planner.add_throw(
        t=0.70,
        ball_id=0,
        p_release=(0.0, 0.0, 0.0),
        duration=1.00,
        p_catch=(0.0, 0.0, 0.0),
        v_hand=(0.0, 0.0, 6.0),
    )

    # Catch event at t=1.70 (required by ball builder)
    planner.add_catch(t=1.70, ball_id=0, p_xyz=(0.0, 0.0, 0.0), v_hand=(0.0, 0.0, 0.0))

    sim = planner.simulate(dt=0.005)

    fig_ts = planner.plot_timeseries(sim)
    # place_figure(fig_ts, x=1920, y=900, w=1000, h=900)

    fig_anim, ani = planner.animate(sim, stride=2, trail=250)
    # place_figure(fig_anim, x=1920, y=0, w=1000, h=900)

    plt.show()


if __name__ == "__main__":
    main()
