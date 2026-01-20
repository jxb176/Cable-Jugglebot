#!/usr/bin/env python3
"""
manual_juggle_planner.py

Manual juggling "pattern editor" (code-based for now):
- Each Ball has a schedule of segments: HELD (hand carries ball) and FLIGHT (ballistic).
- You manually add THROW/CATCH commands with coordinates and throw duration.
- The tool simulates ball motion and shows a 3D animation with play/pause.

Keys:
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
import math

# ----------------------------
# Plotting helper
# ----------------------------

def place_figure(fig, x: int, y: int, w: int, h: int):
    """
    Move/resize a matplotlib figure window in screen pixels.
    Works with Qt and Tk backends (Windows-friendly).
    """
    mgr = fig.canvas.manager

    try:
        # Qt backend
        mgr.window.setGeometry(x, y, w, h)
    except Exception:
        try:
            # Tk backend
            mgr.window.wm_geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass  # backend doesn't support manual placement


# ----------------------------
# Core math
# ----------------------------

def ballistic_v0(p_release: np.ndarray, p_catch: np.ndarray, T: float, g: np.ndarray) -> np.ndarray:
    """
    Solve v0 for ballistic flight:
      p(T) = p0 + v0*T + 0.5*g*T^2  =>  v0 = (p(T) - p0 - 0.5*g*T^2)/T
    Matches the helper in your 2-ball fountain script.
    """
    if T <= 0:
        raise ValueError("Throw duration T must be > 0")
    return (p_catch - p_release - 0.5 * g * (T**2)) / T


def ballistic_state(p0: np.ndarray, v0: np.ndarray, g: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (p(t), v(t)) for ballistic motion under constant gravity."""
    p = p0 + v0 * t + 0.5 * g * (t**2)
    v = v0 + g * t
    return p, v

def smoothstep5(u: float) -> float:
    """Quintic smoothstep: 0->1 with zero vel/acc at endpoints."""
    u = float(np.clip(u, 0.0, 1.0))
    return 10*u**3 - 15*u**4 + 6*u**5

def quintic_coeffs(p0, v0, a0, p1, v1, a1, T) -> np.ndarray:
    c0 = p0
    c1 = v0
    c2 = 0.5 * a0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    A = np.array([[T3, T4, T5],
                  [3*T2, 4*T3, 5*T4],
                  [6*T, 12*T2, 20*T3]], dtype=float)
    b = np.array([p1 - (c0 + c1*T + c2*T2),
                  v1 - (c1 + 2*c2*T),
                  a1 - (2*c2)], dtype=float)
    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4, c5], dtype=float)

def eval_quintic_full(c: np.ndarray, t: float):
    c0, c1, c2, c3, c4, c5 = c
    p = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
    v = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    a = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
    j = 6*c3 + 24*c4*t + 60*c5*t**2
    return p, v, a, j

# ----------------------------
# Data model
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

    # HELD: p0 -> p1 linearly (placeholder until robot planner exists)
    p0: Optional[np.ndarray] = None
    p1: Optional[np.ndarray] = None

    # FLIGHT: ballistic initial conditions
    p_release: Optional[np.ndarray] = None
    v_release: Optional[np.ndarray] = None


class Ball:
    def __init__(self, ball_id: int, radius: float = 0.03):
        self.ball_id = int(ball_id)
        self.radius = float(radius)
        self.segments: List[Segment] = []

    def build_from_events(self, events: List[Event], g: np.ndarray):
        """
        Build alternating segments from CATCH/THROW events.
        Convention:
          - CATCH at time t: ball is (now) held at p
          - THROW at time t: ball leaves hand from p, flies for duration to p_catch
        """
        ev = [e for e in events if e.ball_id == self.ball_id]
        ev = sorted(ev, key=lambda e: e.t)

        if not ev:
            raise ValueError(f"Ball {self.ball_id}: no events")

        expanded = sorted(ev, key=lambda e: e.t)

        # Now build segments:
        # Between CATCH->THROW: HELD (hand carries ball)
        # THROW->CATCH: FLIGHT
        self.segments = []
        i = 0
        while i < len(expanded) - 1:
            a = expanded[i]
            b = expanded[i + 1]
            ka = a.kind.upper()
            kb = b.kind.upper()

            if b.t <= a.t:
                raise ValueError(f"Ball {self.ball_id}: non-increasing times at {a.t} -> {b.t}")

            if ka == "CATCH" and kb == "THROW":
                # HELD segment (placeholder: linear hand path)
                self.segments.append(Segment(
                    t0=a.t, t1=b.t, kind="HELD",
                    p0=a.p.copy(), p1=b.p.copy()
                ))
                i += 1
                continue

            if ka == "THROW":
                if a.duration is None:
                    raise ValueError(f"Ball {self.ball_id}: THROW at t={a.t} missing duration")

                if a.p_catch is None:
                    raise ValueError(f"Ball {self.ball_id}: THROW at t={a.t} missing p_catch")

                t_end = a.t + a.duration

                # Require the next event to be a CATCH exactly at t_end (within epsilon)
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

            # Allow consecutive catches or throws if user does odd things; treat as HELD.
            self.segments.append(Segment(
                t0=a.t, t1=b.t, kind="HELD",
                p0=a.p.copy(), p1=b.p.copy()
            ))
            i += 1

    def state_at(
            self,
            t: float,
            g: np.ndarray,
            hand_eval,  # callable: hand_eval(t)->(p,v,a,j)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (p,v,a,j) at time t.
        HELD: follow the hand exactly.
        FLIGHT: ballistic (a=g, j=0).
        """
        if not self.segments:
            z = np.zeros(3)
            return z, z, z, z

        # clamp outside time range
        if t <= self.segments[0].t0:
            s0 = self.segments[0]
            if s0.kind == "HELD":
                return hand_eval(t)
            else:
                p = s0.p_release.copy()
                v = s0.v_release.copy()
                return p, v, g.copy(), np.zeros(3)

        if t >= self.segments[-1].t1:
            sl = self.segments[-1]
            if sl.kind == "HELD":
                return hand_eval(t)
            else:
                p_end, v_end = ballistic_state(sl.p_release, sl.v_release, g, sl.t1 - sl.t0)
                return p_end, v_end, g.copy(), np.zeros(3)

        # find segment containing t
        for s in self.segments:
            if s.t0 <= t <= s.t1:
                if s.kind == "HELD":
                    return hand_eval(t)
                else:
                    dt = t - s.t0
                    p, v = ballistic_state(s.p_release, s.v_release, g, dt)
                    return p, v, g.copy(), np.zeros(3)

        z = np.zeros(3)
        return z, z, z, z


@dataclass
class HandWaypoint:
    t: float
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray

@dataclass
class HandSeg:
    t0: float
    t1: float
    C: np.ndarray  # (3,6) quintic coeffs per axis

class Hand:
    def __init__(self):
        self.waypoints: List[HandWaypoint] = []
        self.segs: List[HandSeg] = []

    def build_from_events(self, events: List[Event]):
        wps = []
        for e in events:
            k = e.kind.upper()
            if k in ("CATCH", "THROW"):
                v = e.v_hand if e.v_hand is not None else np.zeros(3)
                a = np.zeros(3)
                wps.append(HandWaypoint(t=float(e.t), p=e.p.copy(), v=v.copy(), a=a.copy()))

        wps.sort(key=lambda w: w.t)

        eps = 1e-9
        for i in range(len(wps) - 1):
            if wps[i+1].t <= wps[i].t + eps:
                raise ValueError(f"Hand: non-increasing waypoint times at {wps[i].t} -> {wps[i+1].t}")

        self.waypoints = wps

        # Build per-segment quintics
        self.segs = []
        for a_wp, b_wp in zip(wps[:-1], wps[1:]):
            T = b_wp.t - a_wp.t
            C = np.zeros((3, 6))
            for k in range(3):
                C[k, :] = quintic_coeffs(
                    a_wp.p[k], a_wp.v[k], a_wp.a[k],
                    b_wp.p[k], b_wp.v[k], b_wp.a[k],
                    T
                )
            self.segs.append(HandSeg(t0=a_wp.t, t1=b_wp.t, C=C))

    def eval_at(self, t: float):
        if not self.segs:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        if t <= self.segs[0].t0:
            p = self.waypoints[0].p.copy()
            v = self.waypoints[0].v.copy()
            a = self.waypoints[0].a.copy()
            return p, v, a, np.zeros(3)

        if t >= self.segs[-1].t1:
            p = self.waypoints[-1].p.copy()
            v = self.waypoints[-1].v.copy()
            a = self.waypoints[-1].a.copy()
            return p, v, a, np.zeros(3)

        for s in self.segs:
            if s.t0 <= t <= s.t1:
                tl = t - s.t0
                p = np.zeros(3); v = np.zeros(3); a = np.zeros(3); j = np.zeros(3)
                for k in range(3):
                    p[k], v[k], a[k], j[k] = eval_quintic_full(s.C[k, :], tl)
                return p, v, a, j

        return self.waypoints[-1].p.copy(), np.zeros(3), np.zeros(3), np.zeros(3)


@dataclass
class HandSegSampled:
    t0: float
    t1: float
    T: np.ndarray         # (N,) time samples
    P: np.ndarray         # (N,3)
    V: np.ndarray         # (N,3)
    A: np.ndarray         # (N,3)
    J: np.ndarray         # (N,3)

class HandSCurveVertical:
    """
    Envelope-safe vertical motion during HELD segments:
      - x,y fixed at catch/throw x,y
      - z does a downstroke then upstroke within [z_min+margin, z_max-margin]
    Outside HELD segments: hold last pose with zero derivatives (for now).
    """
    def __init__(self, ws: WorkspaceCylinder, margin: float = 0.01, dip: float = 0.05):
        self.ws = ws
        self.margin = float(margin)
        self.dip = float(dip)
        self.segs: List[HandSegSampled] = []
        self._last_p = np.zeros(3)

    def build_from_events(self, events: List[Event], g: np.ndarray, dt: float = 0.005):
        # Import primitives from your scurve file (same folder)
        from path_playground_gui_scurve_basic_3d import generate_scurve_normalized, map_scurve_to_3d

        # Build HELD intervals for *any* ball: CATCH->THROW defines when hand is holding that ball.
        # For now (single ball test), we’ll just take the first ball’s held intervals.
        ev = sorted([e for e in events if e.kind.upper() in ("CATCH", "THROW")], key=lambda e: e.t)

        # Extract CATCH->THROW pairs per ball_id (simple and robust)
        by_ball: Dict[int, List[Event]] = {}
        for e in ev:
            by_ball.setdefault(e.ball_id, []).append(e)

        held_pairs = []
        for bid, lst in by_ball.items():
            lst.sort(key=lambda e: e.t)
            for a, b in zip(lst[:-1], lst[1:]):
                if a.kind.upper() == "CATCH" and b.kind.upper() == "THROW":
                    held_pairs.append((a, b))

        held_pairs.sort(key=lambda ab: ab[0].t)
        self.segs = []

        zmin = self.ws.z_min + self.margin
        zmax = self.ws.z_max - self.margin

        # Hand motion limits (physical units). Tune later.
        # These are limits on the *hand* along the vertical stroke.
        vz_max = 10.0     # m/s
        az_max = 100.0    # m/s^2
        jz_max = 2000.0   # m/s^3
        sample_hz = 400.0

        for catch_ev, throw_ev in held_pairs:
            t0 = float(catch_ev.t)
            t1 = float(throw_ev.t)
            if t1 <= t0:
                continue

            # Keep x,y at the catch location (vertical test assumption)
            p0 = catch_ev.p.copy()
            p_base = p0.copy()

            # Dip (temp, replace with planning of this)
            # Endpoint vertical velocities (from autofill or manual)
            v0z = float(catch_ev.v_hand[2]) if catch_ev.v_hand is not None else 0.0
            v1z = float(throw_ev.v_hand[2]) if throw_ev.v_hand is not None else 0.0

            v0z = float(np.clip(v0z, -vz_max, vz_max))
            v1z = float(np.clip(v1z, -vz_max, vz_max))

            # --- Auto-size dip based on required endpoint speeds ---
            z0 = float(p_base[2])

            v_need = max(abs(v0z), abs(v1z))  # m/s

            # Minimum distance to change speed by v_need with accel limit (constant-accel lower bound)
            dip_min = (v_need * v_need) / (2.0 * az_max + 1e-12)

            # Safety factor because jerk-limited profiles need more distance
            dip_req = max(self.dip, 1.8 * dip_min)

            # Clamp dip so we can stay inside the z bounds
            max_down = z0 - zmin
            max_up = zmax - z0

            # Prefer dipping downward; if not possible, dip upward
            if max_down >= dip_req:
                z_mid = z0 - dip_req
            elif max_up >= dip_req:
                z_mid = z0 + dip_req
            else:
                # Choose the larger available direction (may still fail later with a clear error)
                z_mid = z0 - max_down if max_down >= max_up else z0 + max_up

            z_mid = float(np.clip(z_mid, zmin, zmax))

            p_mid = p_base.copy()
            p_mid[2] = z_mid

            # ---- Build downstroke (catch -> mid), end at 0 vel
            dz_down = float(p_mid[2] - p_base[2])
            L_down = abs(dz_down)
            if L_down < 1e-9:
                # no dip possible; fall back to holding
                continue

            # Convert endpoint velocities & limits into s-space (s in [0,1])

            # Correct projection: along-path speed must be positive
            sdot0 = (v0z * np.sign(dz_down)) / L_down

            sdot1 = 0.0
            sdot_max = vz_max / L_down
            amax_s = az_max / L_down
            jmax_s = jz_max / L_down

            print(
                f"[SCURVE DOWN] L_down={L_down:.4f}  v0z={v0z:.4f}  "
                f"sdot0={sdot0:.4f}  sdot_max={sdot_max:.4f}  "
                f"amax_s={amax_s:.4f}  jmax_s={jmax_s:.4f}"
            )

            sc_down, _info_down = generate_scurve_normalized(
                sdot0=sdot0, sdot1=sdot1,
                sdot_max=sdot_max,
                amax=amax_s,
                jmax=jmax_s,
                sample_hz=sample_hz,
            )
            traj_down = map_scurve_to_3d(sc_down, p_base, p_mid)

            # ---- Build upstroke (mid -> throw point), start at 0 vel, end at v1z
            dz_up = float(p_base[2] - p_mid[2])
            L_up = abs(dz_up)
            if L_up < 1e-9:
                continue

            sdot0 = 0.0

            # Path directions
            dz_up = p_base[2] - p_mid[2]  # positive
            # Correct projection: along-path speed must be positive
            sdot1 = (v1z * np.sign(dz_up)) / L_up

            sdot_max = vz_max / L_up
            amax_s = az_max / L_up
            jmax_s = jz_max / L_up

            print(
                f"[SCURVE UP] L_up={L_up:.4f}  v1z={v1z:.4f}  "
                f"sdot1={sdot1:.4f}  sdot_max={sdot_max:.4f}  "
                f"amax_s={amax_s:.4f}  jmax_s={jmax_s:.4f}"
            )

            sc_up, _info_up = generate_scurve_normalized(
                sdot0=sdot0, sdot1=sdot1,
                sdot_max=sdot_max,
                amax=amax_s,
                jmax=jmax_s,
                sample_hz=sample_hz,
            )
            traj_up = map_scurve_to_3d(sc_up, p_mid, p_base)

            # ---- Time-align to the HELD interval [t0, t1]
            t_down = float(traj_down[-1, 0])
            t_up = float(traj_up[-1, 0])
            held_T = (t1 - t0)

            if t_down + t_up > held_T + 1e-9:
                raise ValueError(
                    f"Hand HELD interval too short for S-curve limits: "
                    f"need {t_down + t_up:.3f}s but have {held_T:.3f}s "
                    f"(ball {catch_ev.ball_id}, catch {t0:.3f} -> throw {t1:.3f}). "
                    f"Increase hold time or relax vz/az/jz limits."
                )

            pad_T = held_T - (t_down + t_up)  # idle time at the bottom

            print(f"[HELD] held_T={held_T:.3f}  t_down={t_down:.3f}  t_up={t_up:.3f}  pad={pad_T:.3f}")

            # Shift to absolute time
            traj_down[:, 0] += t0

            # Optional dwell at mid (zero v/a/j)
            if pad_T > 1e-12:
                t_hold = np.arange(0.0, pad_T + 1e-12, dt)
                hold = np.zeros((len(t_hold), 13), dtype=float)
                hold[:, 0] = t_hold + (t0 + t_down)
                hold[:, 1:4] = p_mid[None, :]
                # v/a/j already zero
            else:
                hold = np.zeros((0, 13), dtype=float)

            traj_up[:, 0] += (t0 + t_down + pad_T)

            traj = np.vstack([traj_down, hold, traj_up])


            # Resample onto your simulator dt grid for clean eval_at
            T = np.arange(t0, t1 + 1e-12, dt)
            P = np.column_stack([np.interp(T, traj[:,0], traj[:,1]),
                                 np.interp(T, traj[:,0], traj[:,2]),
                                 np.interp(T, traj[:,0], traj[:,3])])
            V = np.column_stack([np.interp(T, traj[:,0], traj[:,4]),
                                 np.interp(T, traj[:,0], traj[:,5]),
                                 np.interp(T, traj[:,0], traj[:,6])])
            A = np.column_stack([np.interp(T, traj[:,0], traj[:,7]),
                                 np.interp(T, traj[:,0], traj[:,8]),
                                 np.interp(T, traj[:,0], traj[:,9])])
            J = np.column_stack([np.interp(T, traj[:,0], traj[:,10]),
                                 np.interp(T, traj[:,0], traj[:,11]),
                                 np.interp(T, traj[:,0], traj[:,12])])

            self.segs.append(HandSegSampled(t0=t0, t1=t1, T=T, P=P, V=V, A=A, J=J))
            self._last_p = p_base.copy()

    def scurve_min_distance_for_dv(dv: float, amax: float, jmax: float) -> float:
        """
        Minimum distance to change speed by dv (>=0) using jerk-limited accel profile:
        - start/end acceleration = 0
        - |a| <= amax, |j| <= jmax
        """
        dv = abs(float(dv))
        amax = float(amax)
        jmax = float(jmax)
        if dv <= 1e-15:
            return 0.0
        if amax <= 0 or jmax <= 0:
            raise ValueError("amax and jmax must be > 0")

        dv_boundary = (amax * amax) / jmax  # where you just hit amax

        if dv <= dv_boundary + 1e-15:
            # triangular accel (jerk up, jerk down)
            return (dv ** 1.5) / math.sqrt(jmax)

        # trapezoidal accel (jerk up, hold accel, jerk down)
        return dv * (amax * amax + jmax * dv) / (2.0 * amax * jmax)

    def eval_at(self, t: float):
        if not self.segs:
            z = np.zeros(3)
            return z, z, z, z

        # Find segment containing t
        for s in self.segs:
            if s.t0 <= t <= s.t1:
                # nearest index (sampled grid)
                i = int(np.clip(np.searchsorted(s.T, t), 0, len(s.T)-1))
                return s.P[i].copy(), s.V[i].copy(), s.A[i].copy(), s.J[i].copy()

        # Outside HELD: hold last pose
        p = self._last_p.copy()
        z = np.zeros(3)
        return p, z, z, z


# ----------------------------
# Planner / Simulator
# ----------------------------

@dataclass
class WorkspaceCylinder:
    radius: float = 0.25
    z_min: float = -0.25
    z_max: float = 0.25


class ManualJugglingPlanner:
    def __init__(self, g_mag: float = 9.81, ws: Optional[WorkspaceCylinder] = None):
        self.g = np.array([0.0, 0.0, -abs(g_mag)], dtype=float)
        self.ws = ws or WorkspaceCylinder()
        self.events: List[Event] = []
        self.balls: Dict[int, Ball] = {}
        self.hand = HandSCurveVertical(self.ws, margin=0.01, dip=0.05)


    def get_ball(self, ball_id: int) -> Ball:
        if ball_id not in self.balls:
            self.balls[ball_id] = Ball(ball_id=ball_id)
        return self.balls[ball_id]

    def add_catch(
            self,
            t: float,
            ball_id: int,
            p_xyz: Tuple[float, float, float],
            vel_scale: float = 1.0,
            v_hand: Optional[Tuple[float, float, float]] = None,
    ):
        self.events.append(Event(
            t=float(t), kind="CATCH", ball_id=int(ball_id),
            p=np.array(p_xyz, dtype=float),
            v_hand=None if v_hand is None else np.array(v_hand, dtype=float),
            vel_scale=float(vel_scale),
        ))

    def add_throw(
            self,
            t: float,
            ball_id: int,
            p_release: Tuple[float, float, float],
            duration: float,
            p_catch: Tuple[float, float, float],
            vel_scale: float = 1.0,
            v_hand: Optional[Tuple[float, float, float]] = None,
    ):
        self.events.append(Event(
            t=float(t), kind="THROW", ball_id=int(ball_id),
            p=np.array(p_release, dtype=float),
            duration=float(duration),
            p_catch=np.array(p_catch, dtype=float),
            v_hand=None if v_hand is None else np.array(v_hand, dtype=float),
            vel_scale=float(vel_scale),
        ))

    def _autofill_hand_velocities(self):
        # group events per ball and time-sort
        by_ball: Dict[int, List[Event]] = {}
        for e in self.events:
            by_ball.setdefault(e.ball_id, []).append(e)

        for bid, evs in by_ball.items():
            evs.sort(key=lambda e: e.t)

            for i in range(len(evs)):
                e = evs[i]
                k = e.kind.upper()

                # If user explicitly provided v_hand, keep it.
                if e.v_hand is not None:
                    continue

                if k == "THROW":
                    if e.duration is None or e.p_catch is None:
                        raise ValueError(f"Ball {bid}: THROW at t={e.t} missing duration/p_catch")

                    v_release = ballistic_v0(e.p, e.p_catch, e.duration, self.g)
                    e.v_hand = float(e.vel_scale) * v_release

                elif k == "CATCH":
                    # Default catch velocity: if previous is a THROW that lands now, use its impact vel.
                    if i > 0:
                        prev = evs[i - 1]
                        if prev.kind.upper() == "THROW" and prev.duration is not None and prev.p_catch is not None:
                            t_end = prev.t + prev.duration
                            if abs(e.t - t_end) <= 1e-9:
                                v_release = ballistic_v0(prev.p, prev.p_catch, prev.duration, self.g)
                                v_impact = v_release + self.g * prev.duration
                                e.v_hand = float(e.vel_scale) * v_impact
                                continue

                    # Otherwise, default to zero if it doesn't match a prior throw
                    e.v_hand = np.zeros(3, dtype=float)

    def build(self):
        self._autofill_hand_velocities()

        # build all balls from shared event list
        for bid, b in self.balls.items():
            b.build_from_events(self.events, self.g)

        # Hand waypoints from events (manual mode)
        self.hand.build_from_events(self.events, self.g, dt=0.005)



    def simulate(self, dt: float = 0.005) -> Dict:
        if not self.balls:
            raise ValueError("No balls added.")

        self.build()

        t0 = min(e.t for e in self.events)
        t1 = max(e.t + (e.duration or 0.0) for e in self.events)

        ts = np.arange(t0, t1 + 1e-12, dt)

        out = {"t": ts, "balls": {}}

        HP = np.zeros((len(ts), 3))
        HV = np.zeros((len(ts), 3))
        HA = np.zeros((len(ts), 3))
        HJ = np.zeros((len(ts), 3))
        for i, t in enumerate(ts):
            p, v, a, j = self.hand.eval_at(float(t))
            HP[i] = p;
            HV[i] = v;
            HA[i] = a;
            HJ[i] = j
        out["hand"] = {"p": HP, "v": HV, "a": HA, "j": HJ}

        for bid, b in self.balls.items():
            P = np.zeros((len(ts), 3))
            V = np.zeros((len(ts), 3))
            A = np.zeros((len(ts), 3))
            J = np.zeros((len(ts), 3))
            for i, t in enumerate(ts):
                p, v, a, j = b.state_at(float(t), self.g, self.hand.eval_at)
                P[i] = p
                V[i] = v
                A[i] = a
                J[i] = j
            out["balls"][bid] = {"p": P, "v": V, "a": A, "j": J}

        return out

    def animate(self, sim: Dict, stride: int = 1):
        ts = sim["t"][::stride]
        ball_ids = sorted(sim["balls"].keys())

        # Precompute arrays
        P = {bid: sim["balls"][bid]["p"][::stride] for bid in ball_ids}

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Manual Juggling Preview (balls only)")

        # Axis bounds from actual trajectories (hand + all balls)
        H_full = sim["hand"]["p"]  # (N,3)
        all_pts = [H_full]

        for bid in ball_ids:
            all_pts.append(sim["balls"][bid]["p"])

        pts = np.vstack(all_pts)  # (M,3)

        xmin, ymin, zmin = np.min(pts, axis=0)
        xmax, ymax, zmax = np.max(pts, axis=0)

        # Pad a bit so points aren't on the border
        pad = 0.05  # meters (adjust to taste)
        xmin -= pad;
        xmax += pad
        ymin -= pad;
        ymax += pad
        zmin -= pad;
        zmax += pad

        # Optional: keep cubic aspect so geometry isn't distorted
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        half = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)

        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)

        # Draw workspace cylinder wireframe
        R = self.ws.radius
        th = np.linspace(0, 2*np.pi, 80)
        z = np.linspace(self.ws.z_min, self.ws.z_max, 2)
        TH, ZZ = np.meshgrid(th, z)
        XX = R * np.cos(TH)
        YY = R * np.sin(TH)
        ax.plot_wireframe(XX, YY, ZZ, linewidth=0.5, alpha=0.25)

        hand_marker = ax.plot([], [], [], marker="x", linestyle="None")[0]
        hand_trail  = ax.plot([], [], [], linewidth=1.5, alpha=0.8)[0]
        H = sim["hand"]["p"][::stride]

        # Artists per ball
        scatters = {}
        trails = {}
        for bid in ball_ids:
            scatters[bid] = ax.plot([], [], [], marker="o", linestyle="None")[0]
            trails[bid] = ax.plot([], [], [], linewidth=1.0, alpha=0.7)[0]

        time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

        # Playback controls
        state = {"paused": False, "i": 0}

        def set_artists(i: int):
            t = float(ts[i])
            time_text.set_text(f"t = {t:.3f} s")

            hp = H[i]
            hand_marker.set_data([hp[0]], [hp[1]])
            hand_marker.set_3d_properties([hp[2]])

            k0 = max(0, i - 200)
            ht = H[k0:i + 1]
            hand_trail.set_data(ht[:, 0], ht[:, 1])
            hand_trail.set_3d_properties(ht[:, 2])

            for bid in ball_ids:
                p = P[bid][i]
                scatters[bid].set_data([p[0]], [p[1]])
                scatters[bid].set_3d_properties([p[2]])

                # simple trail
                k0 = max(0, i - 200)  # last N points
                tr = P[bid][k0:i+1]
                trails[bid].set_data(tr[:, 0], tr[:, 1])
                trails[bid].set_3d_properties(tr[:, 2])

            return [time_text, hand_marker, hand_trail, *scatters.values(), *trails.values()]


        def on_key(event):
            if event.key == " ":
                state["paused"] = not state["paused"]
            elif event.key == "right":
                state["i"] = min(state["i"] + 1, len(ts) - 1)
                set_artists(state["i"])
                fig.canvas.draw_idle()
            elif event.key == "left":
                state["i"] = max(state["i"] - 1, 0)
                set_artists(state["i"])
                fig.canvas.draw_idle()
            elif event.key == "r":
                state["i"] = 0
            elif event.key == "escape":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

        def update(_frame):
            if not state["paused"]:
                state["i"] = (state["i"] + 1) if state["i"] < len(ts) - 1 else state["i"]
            return set_artists(state["i"])

        ani = FuncAnimation(fig, update, interval=20, blit=False)
        #plt.show()
        return fig, ani

    def plot_hand_timeseries(self, sim: Dict, ball_id: int = 0):
        t = sim["t"]

        HP = sim["hand"]["p"];
        HV = sim["hand"]["v"];
        HA = sim["hand"]["a"];
        HJ = sim["hand"]["j"]

        if ball_id not in sim["balls"]:
            raise ValueError(f"ball_id={ball_id} not found")
        BP = sim["balls"][ball_id]["p"]
        BV = sim["balls"][ball_id]["v"]
        BA = sim["balls"][ball_id]["a"]
        BJ = sim["balls"][ball_id]["j"]

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(11, 9))
        axs[0].set_title(f"Hand vs Ball {ball_id} kinematics")

        # Position
        axs[0].plot(t, HP[:, 0], label="hand x")
        axs[0].plot(t, HP[:, 1], label="hand y")
        axs[0].plot(t, HP[:, 2], label="hand z")
        axs[0].plot(t, BP[:, 0], linestyle="--", label="ball x")
        axs[0].plot(t, BP[:, 1], linestyle="--", label="ball y")
        axs[0].plot(t, BP[:, 2], linestyle="--", label="ball z")
        axs[0].set_ylabel("pos [m]")
        axs[0].legend(loc="upper right", ncol=2)

        # Velocity
        axs[1].plot(t, HV[:, 0])
        axs[1].plot(t, HV[:, 1])
        axs[1].plot(t, HV[:, 2])
        axs[1].plot(t, BV[:, 0], linestyle="--")
        axs[1].plot(t, BV[:, 1], linestyle="--")
        axs[1].plot(t, BV[:, 2], linestyle="--")
        axs[1].set_ylabel("vel [m/s]")

        # Acceleration
        axs[2].plot(t, HA[:, 0])
        axs[2].plot(t, HA[:, 1])
        axs[2].plot(t, HA[:, 2])
        axs[2].plot(t, BA[:, 0], linestyle="--")
        axs[2].plot(t, BA[:, 1], linestyle="--")
        axs[2].plot(t, BA[:, 2], linestyle="--")
        axs[2].set_ylabel("acc [m/s²]")

        # Jerk
        axs[3].plot(t, HJ[:, 0])
        axs[3].plot(t, HJ[:, 1])
        axs[3].plot(t, HJ[:, 2])
        axs[3].plot(t, BJ[:, 0], linestyle="--")
        axs[3].plot(t, BJ[:, 1], linestyle="--")
        axs[3].plot(t, BJ[:, 2], linestyle="--")
        axs[3].set_ylabel("jerk [m/s³]")
        axs[3].set_xlabel("time [s]")

        fig.tight_layout()
        return fig


# ----------------------------
# Example usage (start here)
# ----------------------------

def main():
    planner = ManualJugglingPlanner(
        g_mag=9.81,
        ws=WorkspaceCylinder(radius=0.25, z_min=-1.0, z_max=1.0),
    )

    # ----- Single-ball self throw/catch at origin -----
    # Start held at origin
    planner.get_ball(0)
    planner.add_catch(t=0.00, ball_id=0, p_xyz=(0.0, 0.0, 0.0))

    # Throw at t=0.20s from origin, catch at same point after 1.0s of flight
    planner.add_throw(t=0.7, ball_id=0, p_release=(0.0, 0.0, 0.0), duration=1.00, p_catch=(0.0, 0.0, 0.0))

    # Optional: keep holding after catch
    planner.add_catch(t=1.7, ball_id=0, p_xyz=(0.0, 0.0, 0.0))

    sim = planner.simulate(dt=0.005)
    fig_ts = planner.plot_hand_timeseries(sim)
    place_figure(fig_ts, x=1920, y=900, w=1000, h=900)
    fig_anim, ani = planner.animate(sim, stride=2)
    place_figure(fig_anim, x=1920, y=0, w=1000, h=900)

    plt.show()

if __name__ == "__main__":
    main()
