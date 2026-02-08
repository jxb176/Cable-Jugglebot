#!/usr/bin/env python3
"""
two_ball_one_hand_juggle.py

First working implementation:
- Generates a 2-ball / 1-hand "fountain" juggling event schedule (catch + release each beat)
- Computes ballistic throw velocities to make each throw land 2 beats later
- Builds a smooth hand trajectory using piecewise quintic segments (pos+vel constrained at waypoints, zero accel endpoints)
- Exports:
    1) events.csv  (time, type, ball_id, x,y,z, vx,vy,vz)
    2) traj.csv    (time, x,y,z, vx,vy,vz, ax,ay,az) sampled at servo rate

Assumptions:
- One "hand" point in space (no orientation yet; angles held at 0)
- Ball flight is ballistic with constant gravity
- Catch/release points are on/near a cylinder centerline, optional gentle orbit in XY
- Catch velocity matching is enabled by default (can be disabled)

Usage examples:
  python two_ball_one_hand_juggle.py --beats 40 --beat-period 0.45 --dwell 0.08 --z-catch 0.15 --height 0.25
  python two_ball_one_hand_juggle.py --r-catch 0.05 --dphi-deg 10

Notes:
- This is intentionally conservative and "debuggable": start with r_catch=0.0.
- If you see infeasible speeds/accels, increase beat-period, reduce height, or disable catch velocity matching.
"""

from __future__ import annotations
import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class Event:
    t: float
    kind: str       # "CATCH" or "RELEASE"
    ball_id: int
    p: np.ndarray   # shape (3,)
    v: np.ndarray   # shape (3,)


@dataclass
class Waypoint:
    t: float
    p: np.ndarray   # shape (3,)
    v: np.ndarray   # shape (3,)


def rotation_orbit_pos(k: int, r_catch: float, z_catch: float, phi0: float, dphi: float) -> np.ndarray:
    """Nominal catch/release position at beat index k."""
    phi = phi0 + k * dphi
    return np.array([r_catch * math.cos(phi), r_catch * math.sin(phi), z_catch], dtype=float)


def ballistic_v0(p_release: np.ndarray, p_catch: np.ndarray, T: float, g: np.ndarray) -> np.ndarray:
    """Compute initial velocity to go from p_release at t=0 to p_catch at t=T under constant gravity g."""
    # p(T) = p0 + v0*T + 0.5*g*T^2  =>  v0 = (p(T) - p0 - 0.5*g*T^2)/T
    return (p_catch - p_release - 0.5 * g * (T**2)) / T


def quintic_coeffs(p0, v0, a0, p1, v1, a1, T) -> np.ndarray:
    """
    Quintic polynomial coefficients for x(t) over t in [0, T]:
      x(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
    satisfying position/velocity/acceleration at both ends.
    """
    c0 = p0
    c1 = v0
    c2 = 0.5 * a0

    # Solve for c3..c5
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5

    A = np.array([
        [   T3,    T4,     T5],
        [ 3*T2,  4*T3,   5*T4],
        [ 6*T,  12*T2,  20*T3],
    ], dtype=float)

    b = np.array([
        p1 - (c0 + c1*T + c2*T2),
        v1 - (c1 + 2*c2*T),
        a1 - (2*c2),
    ], dtype=float)

    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4, c5], dtype=float)


def eval_quintic(c: np.ndarray, t: float) -> Tuple[float, float, float]:
    """Return (pos, vel, acc) at time t for quintic coefficients c0..c5."""
    c0, c1, c2, c3, c4, c5 = c
    pos = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
    vel = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    acc = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
    return pos, vel, acc

def is_catch_waypoint(i: int) -> bool:
    # With your current generation: waypoints alternate CATCH, RELEASE, CATCH, RELEASE...
    # and start with CATCH at i=0.
    return (i % 2) == 0

def is_release_waypoint(i: int) -> bool:
    return not is_catch_waypoint(i)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def build_waypoints_with_optimized_z_and_catch_vz(
    base_waypoints,
    events,
    z_vars: np.ndarray,
    vz_catch_vars: np.ndarray,
    durations: np.ndarray,
    g_mag: float,
    beta_catch: float = 0.75,
):
    """
    Returns a NEW waypoint list with:
      - p.z from z_vars for ALL waypoints
      - v.z at CATCH from vz_catch_vars (free)
      - v at RELEASE computed from ballistic requirement to land at next CATCH for that ball (hard)

    Assumes x/y and vx/vy come from base_waypoints (start simple).
    """
    g = np.array([0.0, 0.0, -abs(g_mag)], dtype=float)

    # New absolute times for waypoints
    t_new = np.cumsum([0.0] + list(durations))

    # Start with copies
    wps = []
    for i, w in enumerate(base_waypoints):
        p = w.p.copy()
        v = w.v.copy()
        p[2] = float(z_vars[i])
        wps.append(Waypoint(t=float(t_new[i]), p=p, v=v))

    # Build quick lookup by ball_id of release->catch waypoint indices in this waypoint list
    # We rely on your event list ordering matching the waypoint list ordering.
    # Make list of (wp_index, event_kind, ball_id)
    meta = []
    for i, e in enumerate(sorted(events, key=lambda E: E.t)):
        meta.append((i, e.kind, e.ball_id))

    # Determine flight windows per ball: RELEASE waypoint i -> next CATCH waypoint j for same ball
    by_ball = {}
    for i, kind, bid in meta:
        by_ball.setdefault(bid, []).append((i, kind))
    for bid in by_ball:
        by_ball[bid].sort(key=lambda x: x[0])

    # Apply free catch vz variables
    ci = 0
    for i in range(len(wps)):
        if is_catch_waypoint(i):
            wps[i].v[2] = float(vz_catch_vars[ci])
            ci += 1

    # Compute RELEASE velocities (hard) from ballistics using optimized positions and new timing
    for bid, seq in by_ball.items():
        # seq is list of (wp_idx, kind) in time order
        for k in range(len(seq)):
            i0, kind0 = seq[k]
            if kind0 != "RELEASE":
                continue
            # find next catch
            j = None
            for kk in range(k + 1, len(seq)):
                i1, kind1 = seq[kk]
                if kind1 == "CATCH":
                    j = i1
                    break
            if j is None:
                continue

            t_r = wps[i0].t
            t_c = wps[j].t
            T = t_c - t_r
            if T <= 1e-6:
                continue

            p_r = wps[i0].p
            p_c = wps[j].p

            v0 = (p_c - p_r - 0.5 * g * (T**2)) / T  # ballistic required

            # HARD: hand release velocity equals ball initial velocity
            wps[i0].v = v0.copy()

            # You can also set catch target (soft) reference here if you want access elsewhere:
            # incoming ball velocity at catch:
            v_in = v0 + g * T
            # keep for debugging if desired
            # wps[j].v_target = beta_catch * v_in  # (not in dataclass)
    return wps


def build_segment_coeffs_from_waypoints_with_times(waypoints, durations):
    """
    Same as before, but uses waypoint.p and waypoint.v, and durations define segment times.
    Endpoint accelerations are 0 for now (we can make them variables later).
    """
    seg_coeffs = []
    for i in range(len(waypoints) - 1):
        w0, w1 = waypoints[i], waypoints[i + 1]
        T = float(durations[i])
        C = np.zeros((3, 6), dtype=float)
        for axis in range(3):
            C[axis, :] = quintic_coeffs(
                p0=float(w0.p[axis]),
                v0=float(w0.v[axis]),
                a0=0.0,
                p1=float(w1.p[axis]),
                v1=float(w1.v[axis]),
                a1=0.0,
                T=T,
            )
        seg_coeffs.append(C)
    return seg_coeffs


def eval_piecewise(seg_coeffs, durations, t_query):
    cum = np.cumsum([0.0] + list(durations))
    if t_query <= 0.0:
        idx, tl = 0, 0.0
    elif t_query >= cum[-1]:
        idx, tl = len(durations) - 1, float(durations[-1])
    else:
        idx = int(np.searchsorted(cum, t_query, side="right") - 1)
        tl = float(t_query - cum[idx])

    C = seg_coeffs[idx]
    p = np.zeros(3)
    v = np.zeros(3)
    a = np.zeros(3)
    for axis in range(3):
        pos, vel, acc = eval_quintic(C[axis, :], tl)
        p[axis], v[axis], a[axis] = pos, vel, acc
    return p, v, a

def optimize_hand_with_envelope_and_throw_constraints(
    base_waypoints,
    events,
    durations0,
    cyl_radius,
    z_min,
    z_max,
    g_mag=9.81,
    eps=0.005,
    beta_catch=0.5,
    catch_soft_weight=5.0,
    time_weight=0.05,
    dev_weight=1.0,
    sample_per_seg=8,
    max_scale=5.0,
    release_v_tol_frac=0.01,  # 1% band if you choose to enforce it as inequality
    use_release_tol_band=False,
    start_at_bottom=False,
):
    """
    Variables:
      - z at every waypoint (Nw)
      - vz at every catch waypoint (Nc)
      - time scale s_j for each segment (Ns)

    Hard:
      - RELEASE velocity set from ballistics exactly (unless you enable tolerance band; see below)

    Soft:
      - Catch velocity should match beta_catch * incoming ball velocity (penalty term)

    Constraints:
      - z(t) within [z_min+eps, z_max-eps] at collocation points
      - (optional) radius within cyl at collocation points (only meaningful if x/y are nonzero)
    """
    durations0 = np.asarray(durations0, dtype=float)
    nseg = len(durations0)
    Nw = len(base_waypoints)
    Nc = (Nw + 1) // 2  # catch waypoints at indices 0,2,4...

    # Initial guesses: use current z and current catch vz
    z0 = np.array([w.p[2] for w in base_waypoints], dtype=float)
    vz_c0 = np.array([base_waypoints[i].v[2] for i in range(Nw) if is_catch_waypoint(i)], dtype=float)
    s0 = np.ones(nseg, dtype=float)

    # Pack/unpack
    def pack(z, vz_c, s):
        return np.concatenate([z, vz_c, s])

    def unpack(x):
        z = x[:Nw]
        vz_c = x[Nw:Nw+Nc]
        s = x[Nw+Nc:]
        return z, vz_c, s

    x0 = pack(z0, vz_c0, s0)

    # Bounds: z within cylinder, catch vz unbounded (or you can bound), s bounded
    bounds = []
    for i in range(Nw):
        if start_at_bottom and i == 0:
            z0_fixed = z_min + eps
            bounds.append((z0_fixed, z0_fixed))
        else:
            bounds.append((z_min + eps, z_max - eps))
    for _ in range(Nc):
        bounds.append((None, None))
    for _ in range(nseg):
        bounds.append((0.4, max_scale))

    fracs = np.linspace(0.0, 1.0, sample_per_seg + 2)[1:-1]
    g = np.array([0.0, 0.0, -abs(g_mag)], dtype=float)

    # We need incoming velocities at catches to build the soft cost.
    # We'll compute them from the RELEASE ballistic solutions each iteration.

    def build_iteration_objects(x):
        z_vars, vz_c_vars, s = unpack(x)
        durations = s * durations0

        wps = build_waypoints_with_optimized_z_and_catch_vz(
            base_waypoints=base_waypoints,
            events=events,
            z_vars=z_vars,
            vz_catch_vars=vz_c_vars,
            durations=durations,
            g_mag=g_mag,
            beta_catch=beta_catch,
        )
        seg_coeffs = build_segment_coeffs_from_waypoints_with_times(wps, durations)
        return wps, durations, seg_coeffs

    def objective(x):
        z_vars, vz_c_vars, s = unpack(x)
        durations = s * durations0
        wps, durations, seg_coeffs = build_iteration_objects(x)

        # Soft catch velocity tracking:
        # For each ball: for each RELEASE->next CATCH, compute incoming velocity at catch and penalize mismatch
        # We only penalize Z for now (easy and effective).
        by_ball = {}
        ev_sorted = sorted(events, key=lambda E: E.t)
        for i, e in enumerate(ev_sorted):
            by_ball.setdefault(e.ball_id, []).append((i, e.kind))
        for b in by_ball:
            by_ball[b].sort(key=lambda x: x[0])

        catch_cost = 0.0
        for b, seq in by_ball.items():
            for k in range(len(seq)):
                i0, kind0 = seq[k]
                if kind0 != "RELEASE":
                    continue
                j = None
                for kk in range(k + 1, len(seq)):
                    i1, kind1 = seq[kk]
                    if kind1 == "CATCH":
                        j = i1
                        break
                if j is None:
                    continue

                t_r = wps[i0].t
                t_c = wps[j].t
                T = t_c - t_r
                if T <= 1e-6:
                    continue

                # Release velocity is set hard in wps[i0].v
                v0 = wps[i0].v
                v_in = v0 + g * T

                v_target_z = beta_catch * v_in[2]
                v_hand_z = wps[j].v[2]  # decision var at catches
                err = v_hand_z - v_target_z
                catch_cost += err * err

        # Time regularization + keep s near 1
        dev = np.sum((s - 1.0) ** 2)
        total_time = np.sum(durations)

        return dev_weight * float(dev) + time_weight * float(total_time) + catch_soft_weight * float(catch_cost)

    def constraints_ineq(x):
        # Return array that must be >= 0
        wps, durations, seg_coeffs = build_iteration_objects(x)
        vals = []

        # Collocation along each segment: enforce z bounds and cylinder radius
        t_abs = 0.0
        for i, T in enumerate(durations):
            C = seg_coeffs[i]
            for f in fracs:
                tl = float(f * T)
                p = np.zeros(3)
                for axis in range(3):
                    p[axis] = eval_quintic(C[axis, :], tl)[0]

                r = math.hypot(p[0], p[1])
                vals.append((cyl_radius - eps) - r)
                vals.append(p[2] - (z_min + eps))
                vals.append((z_max - eps) - p[2])
            t_abs += T

        # Optional: 1% tolerance band on release velocity matching (usually unnecessary since it’s hard by construction)
        if use_release_tol_band:
            ev_sorted = sorted(events, key=lambda E: E.t)
            for i, e in enumerate(ev_sorted):
                if e.kind != "RELEASE":
                    continue
                # Evaluate spline velocity at release time (which is waypoint time by construction)
                # Compare to ballistic-required (wps[i].v)
                # Since wps[i].v is set to required, this is mostly a sanity check.
                v_req = wps[i].v
                v_hand = wps[i].v
                # enforce |v_hand - v_req| <= tol * |v_req|
                tol = release_v_tol_frac * max(1e-6, abs(v_req[2]))
                vals.append(tol - abs(v_hand[2] - v_req[2]))

        return np.asarray(vals, dtype=float)

    cons = [{"type": "ineq", "fun": constraints_ineq}]

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-6, "disp": False},
    )

    # SLSQP often improves feasibility but still reports success=False.
    if hasattr(res, "x") and np.all(np.isfinite(res.x)):
        x_opt = res.x
    else:
        x_opt = x0

    wps, durations, seg_coeffs = build_iteration_objects(x_opt)

    info = {
        "success": bool(res.success),
        "message": str(res.message),
        "iter": int(res.nit) if hasattr(res, "nit") else None,
        "total_time_nominal": float(np.sum(durations0)),
        "total_time_opt": float(np.sum(durations)),
        "worst_margin": float(np.min(constraints_ineq(x_opt))),
    }
    return wps, durations, info


# Optimization Helpers
def build_segment_coeffs_from_waypoints(waypoints, durations):
    """
    Build per-segment quintic coeffs for x/y/z.
    durations: list/array of length (N-1)
    Returns:
      coeffs: list of length (N-1), each item is (3,6) array for xyz quintic coeffs
    """
    N = len(waypoints)
    assert len(durations) == N - 1

    seg_coeffs = []
    for i in range(N - 1):
        w0, w1 = waypoints[i], waypoints[i + 1]
        T = float(durations[i])
        if T <= 1e-9:
            raise ValueError("Non-positive segment duration.")

        C = np.zeros((3, 6), dtype=float)
        for axis in range(3):
            C[axis, :] = quintic_coeffs(
                p0=float(w0.p[axis]),
                v0=float(w0.v[axis]),
                a0=0.0,
                p1=float(w1.p[axis]),
                v1=float(w1.v[axis]),
                a1=0.0,
                T=T,
            )
        seg_coeffs.append(C)
    return seg_coeffs


def eval_piecewise_at(seg_coeffs, durations, t_query):
    """
    Evaluate piecewise trajectory at absolute time t_query, where segment boundaries are cumulative sums of durations,
    starting at t=0.
    Returns p,v,a (3,)
    """
    # Find which segment contains t_query
    cum = np.cumsum([0.0] + list(durations))
    if t_query <= 0.0:
        idx = 0
        tl = 0.0
    elif t_query >= cum[-1]:
        idx = len(durations) - 1
        tl = durations[-1]
    else:
        idx = int(np.searchsorted(cum, t_query, side="right") - 1)
        tl = float(t_query - cum[idx])

    C = seg_coeffs[idx]
    p = np.zeros(3)
    v = np.zeros(3)
    a = np.zeros(3)
    for axis in range(3):
        pos, vel, acc = eval_quintic(C[axis, :], tl)
        p[axis], v[axis], a[axis] = pos, vel, acc
    return p, v, a


def compute_throw_mid_times(events, waypoints, durations):
    """
    Build a list of mid-times for each ball throw window based on RELEASE->next CATCH for same ball.
    Times are absolute in the "optimized timeline" where waypoint 0 starts at t=0 and waypoint i boundary times are cumulative sums of durations.

    This assumes events/waypoints are time-ordered and correspond (in order) to those events.
    In our generator, waypoints include both CATCH and RELEASE and are already sorted.
    We'll map event times to waypoint indices by matching (time, kind, ball_id) approximately.
    """
    # Build lookup from (kind, ball_id, t) -> waypoint index
    # Use tolerance because floats
    tol = 1e-9
    wp_meta = []
    for w in waypoints:
        wp_meta.append(w.t)  # original times (pre-optimization)

    # Map old time -> waypoint index (unique in this generator)
    time_to_wpi = {wp_meta[i]: i for i in range(len(wp_meta))}

    # Build cumulative new times of waypoints from durations
    wp_new_t = np.cumsum([0.0] + list(durations))

    # Extract release/catch times by ball in the *original* schedule
    by_ball = {}
    for e in events:
        by_ball.setdefault(e.ball_id, []).append(e)
    for b in by_ball:
        by_ball[b].sort(key=lambda e: e.t)

    mid_times = []
    for b, evs in by_ball.items():
        for i in range(len(evs)):
            if evs[i].kind != "RELEASE":
                continue
            # find next catch
            j = None
            for k in range(i + 1, len(evs)):
                if evs[k].kind == "CATCH":
                    j = k
                    break
            if j is None:
                continue

            t_rel_old = evs[i].t
            t_cat_old = evs[j].t

            # locate nearest waypoint indices
            if t_rel_old not in time_to_wpi or t_cat_old not in time_to_wpi:
                # should not happen in your generator
                continue

            i_rel = time_to_wpi[t_rel_old]
            i_cat = time_to_wpi[t_cat_old]

            t_rel_new = wp_new_t[i_rel]
            t_cat_new = wp_new_t[i_cat]
            mid_times.append(0.5 * (t_rel_new + t_cat_new))

    return mid_times


def optimize_time_allocation_to_fit_cylinder(
    waypoints,
    events,
    durations0,
    cyl_radius,
    z_min,
    z_max,
    eps=0.005,
    sample_per_seg=5,
    add_throw_mid_constraints=True,
    max_scale=4.0,
):
    """
    Optimize segment durations (time allocation) so the resulting quintic spline stays inside cylinder:
      sqrt(x^2+y^2) <= R - eps
      z_min+eps <= z <= z_max-eps

    Keeps waypoint positions/velocities fixed. Only adjusts segment durations.

    Objective: keep durations close to nominal while allowing stretching.
    """
    durations0 = np.asarray(durations0, dtype=float)
    nseg = len(durations0)

    # Variables are scale factors s_i, durations = s_i * durations0
    # Bounds keep them positive and limited.
    s0 = np.ones(nseg, dtype=float)

    bounds = [(0.4, max_scale) for _ in range(nseg)]  # 0.4x .. max_scale

    # Precompute sample fractions in each segment (avoid exactly 0 or 1)
    fracs = np.linspace(0.0, 1.0, sample_per_seg + 2)[1:-1]

    # Weighting: penalize deviation from nominal + total time
    w_dev = 1.0  # how hard we try to keep durations near nominal
    w_time = 0.05  # small bias toward shorter total time

    def objective(s):
        s = np.asarray(s, dtype=float)
        d = s * durations0
        return w_dev * float(np.sum((s - 1.0) ** 2)) + w_time * float(np.sum(d))

    def constraint_values(s):
        """
        Return an array of constraint margins where all must be >= 0.
        Each margin is:
          (R-eps) - r
          z - (z_min+eps)
          (z_max-eps) - z
        sampled at interior points of segments (+ optionally mid-throw times).
        """
        s = np.asarray(s, dtype=float)
        durs = s * durations0
        seg_coeffs = build_segment_coeffs_from_waypoints(waypoints, durs)

        vals = []

        # Segment interior samples
        t0 = 0.0
        for i in range(nseg):
            T = durs[i]
            for f in fracs:
                tl = f * T
                # Evaluate local time on this segment directly (faster than eval_piecewise)
                C = seg_coeffs[i]
                p = np.zeros(3)
                for axis in range(3):
                    p[axis] = eval_quintic(C[axis, :], tl)[0]

                r = math.hypot(p[0], p[1])
                vals.append((cyl_radius - eps) - r)
                vals.append(p[2] - (z_min + eps))
                vals.append((z_max - eps) - p[2])

            t0 += T

        # Mid-throw constraints (optional)
        if add_throw_mid_constraints:
            mid_times = compute_throw_mid_times(events, waypoints, durs)
            for tm in mid_times:
                p, _, _ = eval_piecewise_at(seg_coeffs, durs, tm)
                r = math.hypot(p[0], p[1])
                vals.append((cyl_radius - eps) - r)
                vals.append(p[2] - (z_min + eps))
                vals.append((z_max - eps) - p[2])

        return np.asarray(vals, dtype=float)

    # SLSQP supports inequality constraints g(x) >= 0
    cons = [{"type": "ineq", "fun": constraint_values}]

    res = minimize(
        objective,
        s0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 200, "ftol": 1e-6, "disp": False},
    )

    s_opt = res.x if res.success else s0
    d_opt = s_opt * durations0

    # Report worst constraint after
    cv = constraint_values(s_opt)
    worst = float(np.min(cv))

    info = {
        "success": bool(res.success),
        "message": str(res.message),
        "iter": int(res.nit) if hasattr(res, "nit") else None,
        "worst_margin": worst,
        "scale_min": float(np.min(s_opt)),
        "scale_max": float(np.max(s_opt)),
        "total_time_nominal": float(np.sum(durations0)),
        "total_time_opt": float(np.sum(d_opt)),
    }
    return d_opt, info

def clamp_point_to_cylinder(p: np.ndarray, R: float, z_min: float, z_max: float) -> np.ndarray:
    """Clamp a 3D point into a vertical cylinder of radius R and z in [z_min, z_max]."""
    pc = p.copy()
    pc[2] = float(np.clip(pc[2], z_min, z_max))
    r = math.hypot(pc[0], pc[1])
    if r > R and r > 1e-12:
        scale = R / r
        pc[0] *= scale
        pc[1] *= scale
    return pc


def points_outside_cylinder(p: np.ndarray, R: float, z_min: float, z_max: float) -> bool:
    r = math.hypot(p[0], p[1])
    return (r > R + 1e-9) or (p[2] < z_min - 1e-9) or (p[2] > z_max + 1e-9)

def generate_events_and_waypoints(
    beats: int,
    beat_period: float,
    dwell: float,
    z_catch: float,
    height: float,
    r_catch: float,
    dphi_deg: float,
    phi0_deg: float,
    cyl_radius: float,
    z_min: float,
    z_max: float,
    g_mag: float = 9.81,
    match_catch_velocity: bool = True,
    enforce_waypoints_in_cylinder: bool = True,
) -> Tuple[List[Event], List[Waypoint], dict]:
    """
    Generates:
      - Events (catch and release)
      - Waypoints for the hand trajectory (time, pos, vel)

    Pattern:
      Catch at t_k = k*Δ, Release at t_k + dwell
      Each release lands 2 beats later (same ball re-caught at k+2).
    """
    assert beats >= 3, "Need at least 3 beats to define the 2-beat-later landing."
    assert 0.0 <= dwell < beat_period, "dwell must be in [0, beat_period)."

    Δ = beat_period
    T_flight = 2.0 * Δ - dwell  # time from release at (k*Δ + dwell) to catch at ((k+2)*Δ)

    # Gravity vector (down in +z convention? We'll use z up, gravity negative z.)
    g = np.array([0.0, 0.0, -abs(g_mag)], dtype=float)

    # Orbit parameters
    dphi = math.radians(dphi_deg)
    phi0 = math.radians(phi0_deg)

    # We'll generate beats from k = -2 .. beats (inclusive for final catch),
    # so that catch at k=0 has an incoming velocity from the throw at k=-2.
    k_min = -2
    # Need p_c up to beats+1 because the last release at k=beats-1 lands at k+2 = beats+1
    k_max = beats + 2
    ks = list(range(k_min, k_max + 1))

    # Nominal catch positions for each beat index
    p_c = {}
    for k in ks:
        p_nom = rotation_orbit_pos(k, r_catch, z_catch, phi0, dphi)
        p_clamped = clamp_point_to_cylinder(p_nom, cyl_radius, z_min, z_max)
        if enforce_waypoints_in_cylinder and np.linalg.norm(p_clamped - p_nom) > 1e-6:
            raise ValueError(
                f"Waypoint outside cylinder at beat k={k}: p={p_nom} clamped={p_clamped}. "
                f"Adjust z_catch/r_catch/cylinder bounds."
            )
        p_c[k] = p_clamped

    # OPTIONAL: enforce apex height by adjusting vertical component? (simple and robust)
    # If you want a specific apex above the *release*, the required vz0 is sqrt(2 g h).
    # But the ballistic endpoint constraint already defines vz0; height is used here only as a sanity check.
    # We'll still compute "implied apex" for reporting.
    def implied_apex_z(p0: np.ndarray, v0: np.ndarray) -> float:
        # apex occurs when vz=0: t = -vz0/gz (gz negative)
        gz = g[2]
        vz0 = v0[2]
        if vz0 <= 0.0:
            return p0[2]
        t_apex = -vz0 / gz
        return p0[2] + vz0 * t_apex + 0.5 * gz * (t_apex**2)

    # Throw velocities v0(k): release at beat k, lands at beat k+2
    v0 = {}
    apex_z = {}
    for k in range(k_min, beats):  # releases occur for k up to beats-1 (we'll generate up to beats-1)
        p_rel = p_c[k]
        p_land = p_c[k + 2]
        v = ballistic_v0(p_rel, p_land, T_flight, g)
        v0[k] = v
        apex_z[k] = implied_apex_z(p_rel, v)

    # Incoming catch velocity at beat k comes from throw at k-2
    v_in = {}
    for k in range(0, beats + 1):  # catches up to final beat
        k_throw = k - 2
        if k_throw not in v0:
            # Should only happen very early if you choose fewer beats; handled by assertion above.
            v_in[k] = np.zeros(3)
        else:
            v_in[k] = v0[k_throw] + g * T_flight

    # Build events and waypoints
    events: List[Event] = []
    waypoints: List[Waypoint] = []

    for k in range(0, beats + 1):
        t_k = k * Δ
        ball_id = k % 2  # alternate balls each beat

        # Catch event at t_k
        p_k = p_c[k]
        v_catch = v_in[k] if match_catch_velocity else np.zeros(3)

        events.append(Event(t=t_k, kind="CATCH", ball_id=ball_id, p=p_k, v=v_catch))
        waypoints.append(Waypoint(t=t_k, p=p_k, v=v_catch))

        # Release event at t_k + dwell for beats 0..beats-1
        if k < beats:
            t_rel = t_k + dwell
            v_rel = v0[k]  # throw so it lands at k+2
            events.append(Event(t=t_rel, kind="RELEASE", ball_id=ball_id, p=p_k, v=v_rel))
            waypoints.append(Waypoint(t=t_rel, p=p_k, v=v_rel))

    # Sort by time (catch then release order already correct; but sort anyway)
    events.sort(key=lambda e: e.t)
    waypoints.sort(key=lambda w: w.t)

    # Diagnostics / report
    stats = {
        "T_flight": T_flight,
        "Δ": Δ,
        "dwell": dwell,
        "match_catch_velocity": match_catch_velocity,
        "apex_z_min": float(min(apex_z.values())) if apex_z else float("nan"),
        "apex_z_max": float(max(apex_z.values())) if apex_z else float("nan"),
        "apex_z_mean": float(np.mean(list(apex_z.values()))) if apex_z else float("nan"),
    }

    # Simple warning if implied apex differs a lot from desired z_catch+height
    desired_apex = z_catch + height
    stats["desired_apex_z"] = float(desired_apex)
    stats["apex_error_mean"] = float(stats["apex_z_mean"] - desired_apex)

    return events, waypoints, stats


def build_quintic_trajectory(
    waypoints: List[Waypoint],
    sample_hz: float,
    a0_end: np.ndarray | None = None,
    a1_end: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a sampled trajectory by connecting consecutive waypoints with quintic segments per axis.

    Endpoint accelerations are set to 0 by default (per waypoint, per segment).
    Returns array with columns: t, x,y,z, vx,vy,vz, ax,ay,az
    """
    assert len(waypoints) >= 2
    dt = 1.0 / sample_hz

    # Default endpoint accelerations
    if a0_end is None:
        a0_end = np.zeros(3)
    if a1_end is None:
        a1_end = np.zeros(3)

    rows = []

    for i in range(len(waypoints) - 1):
        w0 = waypoints[i]
        w1 = waypoints[i + 1]
        T = w1.t - w0.t
        if T <= 1e-9:
            continue

        # Quintic coeffs for x,y,z
        coeffs = []
        for axis in range(3):
            c = quintic_coeffs(
                p0=float(w0.p[axis]),
                v0=float(w0.v[axis]),
                a0=float(a0_end[axis]),
                p1=float(w1.p[axis]),
                v1=float(w1.v[axis]),
                a1=float(a1_end[axis]),
                T=float(T),
            )
            coeffs.append(c)

        # Sample segment [0, T) except include final sample on last segment
        n = int(math.floor(T / dt))
        t_local_samples = [j * dt for j in range(n)]
        if i == len(waypoints) - 2:
            # include endpoint exactly
            t_local_samples.append(T)

        for tl in t_local_samples:
            t_abs = w0.t + tl
            p = np.zeros(3)
            v = np.zeros(3)
            a = np.zeros(3)
            for axis in range(3):
                pos, vel, acc = eval_quintic(coeffs[axis], tl)
                p[axis] = pos
                v[axis] = vel
                a[axis] = acc
            rows.append([t_abs, p[0], p[1], p[2], v[0], v[1], v[2], a[0], a[1], a[2]])

    return np.array(rows, dtype=float)

def build_quintic_trajectory_from_durations(waypoints, durations, sample_hz):
    """
    Sample the piecewise quintic trajectory when segment durations are provided explicitly.
    Returns array: t, x,y,z, vx,vy,vz, ax,ay,az
    """
    dt = 1.0 / sample_hz
    seg_coeffs = build_segment_coeffs_from_waypoints(waypoints, durations)

    rows = []
    t_abs = 0.0
    for i, T in enumerate(durations):
        C = seg_coeffs[i]
        n = int(math.floor(T / dt))
        t_locals = [j * dt for j in range(n)]
        if i == len(durations) - 1:
            t_locals.append(T)

        for tl in t_locals:
            p = np.zeros(3)
            v = np.zeros(3)
            a = np.zeros(3)
            for axis in range(3):
                pos, vel, acc = eval_quintic(C[axis, :], tl)
                p[axis], v[axis], a[axis] = pos, vel, acc
            rows.append([t_abs + tl, p[0], p[1], p[2], v[0], v[1], v[2], a[0], a[1], a[2]])

        t_abs += T

    return np.array(rows, dtype=float)


def write_events_csv(path: str, events: List[Event]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "type", "ball_id", "x", "y", "z", "vx", "vy", "vz"])
        for e in events:
            w.writerow([f"{e.t:.6f}", e.kind, e.ball_id,
                        f"{e.p[0]:.6f}", f"{e.p[1]:.6f}", f"{e.p[2]:.6f}",
                        f"{e.v[0]:.6f}", f"{e.v[1]:.6f}", f"{e.v[2]:.6f}"])


def write_traj_csv(path: str, traj: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"])
        for row in traj:
            w.writerow([f"{row[0]:.6f}",
                        f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}",
                        f"{row[4]:.6f}", f"{row[5]:.6f}", f"{row[6]:.6f}",
                        f"{row[7]:.6f}", f"{row[8]:.6f}", f"{row[9]:.6f}"])

def clamp_traj_to_cylinder(traj: np.ndarray, R: float, z_min: float, z_max: float) -> np.ndarray:
    """
    Clamp sampled trajectory positions into cylinder and recompute v/a numerically.
    traj columns: t, x,y,z, vx,vy,vz, ax,ay,az
    """
    t = traj[:, 0].copy()
    p = traj[:, 1:4].copy()

    # Clamp positions
    for i in range(len(p)):
        p[i] = clamp_point_to_cylinder(p[i], R, z_min, z_max)

    # Recompute v and a from clamped positions (finite differences)
    v = np.zeros_like(p)
    a = np.zeros_like(p)
    for axis in range(3):
        v[:, axis] = np.gradient(p[:, axis], t)
        a[:, axis] = np.gradient(v[:, axis], t)

    out = traj.copy()
    out[:, 1:4] = p
    out[:, 4:7] = v
    out[:, 7:10] = a
    return out


#Plotting helpers
def compute_jerk(traj: np.ndarray) -> np.ndarray:
    """
    Compute jerk by differentiating acceleration with respect to time.
    traj columns: t, x,y,z, vx,vy,vz, ax,ay,az
    Returns jerk array shape (N,3)
    """
    t = traj[:, 0]
    a = traj[:, 7:10]
    # Use np.gradient for reasonably stable numerical derivative on uniform-ish time grids.
    j = np.zeros_like(a)
    for i in range(3):
        j[:, i] = np.gradient(a[:, i], t)
    return j


def plot_stacked_kinematics(traj: np.ndarray, show: bool = True, save_path: str | None = None) -> None:
    """
    Stacked plots for position, velocity, acceleration, jerk (XYZ) vs time.
    """
    t = traj[:, 0]
    p = traj[:, 1:4]
    v = traj[:, 4:7]
    a = traj[:, 7:10]
    j = compute_jerk(traj)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    labels = ["x", "y", "z"]

    axes[0].set_title("End Effector (Hand) Kinematics in Global Frame")
    for i in range(3):
        axes[0].plot(t, p[:, i], label=labels[i])
    axes[0].set_ylabel("Position (m)")
    axes[0].grid(True)
    axes[0].legend()

    for i in range(3):
        axes[1].plot(t, v[:, i], label=labels[i])
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True)

    for i in range(3):
        axes[2].plot(t, a[:, i], label=labels[i])
    axes[2].set_ylabel("Accel (m/s²)")
    axes[2].grid(True)

    for i in range(3):
        axes[3].plot(t, j[:, i], label=labels[i])
    axes[3].set_ylabel("Jerk (m/s³)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    if show:
        plt.show()


def build_ball_event_tables(events: List[Event]) -> dict[int, List[Event]]:
    """
    Return {ball_id: [events_sorted_by_time]}.
    """
    by_ball: dict[int, List[Event]] = {}
    for e in events:
        by_ball.setdefault(e.ball_id, []).append(e)
    for bid in by_ball:
        by_ball[bid].sort(key=lambda e: e.t)
    return by_ball


def ball_position_at_time(
    ball_id: int,
    t: float,
    events_by_ball: dict[int, List[Event]],
    hand_t: np.ndarray,
    hand_p: np.ndarray,
    g: np.ndarray,
) -> np.ndarray:
    """
    Ball position model:
    - If last event before t is RELEASE and next is CATCH: ballistic flight
    - Otherwise (in hand / hold): follow hand position by interpolating from sampled hand trajectory

    hand_t: time array (N,)
    hand_p: position array (N,3)
    """
    # Find the last event at or before t
    evs = events_by_ball[ball_id]
    idx = None
    for i in range(len(evs) - 1, -1, -1):
        if evs[i].t <= t:
            idx = i
            break

    # If no event yet, just follow hand (safe fallback)
    if idx is None:
        return interp_hand_pos(t, hand_t, hand_p)

    e_last = evs[idx]
    e_next = evs[idx + 1] if idx + 1 < len(evs) else None

    if e_last.kind == "RELEASE" and (e_next is not None and e_next.kind == "CATCH") and t < e_next.t:
        # Ballistic
        dt = t - e_last.t
        return e_last.p + e_last.v * dt + 0.5 * g * (dt**2)

    # Otherwise: in hand / held
    return interp_hand_pos(t, hand_t, hand_p)


def interp_hand_pos(tq: float, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Linear interpolate hand position at tq given sampled (t, p).
    """
    if tq <= t[0]:
        return p[0]
    if tq >= t[-1]:
        return p[-1]
    i = np.searchsorted(t, tq) - 1
    i = max(0, min(i, len(t) - 2))
    t0, t1 = t[i], t[i + 1]
    w = (tq - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
    return (1 - w) * p[i] + w * p[i + 1]


def animate_juggling_3d(
    traj: np.ndarray,
    events: List[Event],
    workspace_radius: float = 0.25,
    z_min: float = -0.25,
    z_max: float = 0.25,
    g_mag: float = 9.81,
    stride: int = 2,
    save_mp4: str | None = None,
) -> None:
    """
    3D animation: hand + two balls.
    - stride: decimation factor for frames (use 2..10 to speed up)
    - save_mp4: if provided, attempts to save using ffmpeg.
    """
    t = traj[:, 0]
    hand_p = traj[:, 1:4]

    g = np.array([0.0, 0.0, -abs(g_mag)], dtype=float)
    events_by_ball = build_ball_event_tables(events)

    # Frame indices
    frame_idxs = np.arange(0, len(t), stride)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("2-Ball / 1-Hand Juggling Simulation")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Set limits (cylindrical workspace visual approx)
    ax.set_xlim(-workspace_radius, workspace_radius)
    ax.set_ylim(-workspace_radius, workspace_radius)
    ax.set_zlim(z_min, z_max)

    # Draw a faint cylinder wireframe (workspace)
    theta = np.linspace(0, 2 * np.pi, 40)
    zline = np.linspace(z_min, z_max, 20)
    TH, ZZ = np.meshgrid(theta, zline)
    XX = workspace_radius * np.cos(TH)
    YY = workspace_radius * np.sin(TH)
    ax.plot_wireframe(XX, YY, ZZ, rstride=4, cstride=6, linewidth=0.5, alpha=0.25)

    # Artists: hand + 2 balls + optional traces
    hand_scatter = ax.scatter([], [], [], s=60)
    ball0_scatter = ax.scatter([], [], [], s=40)
    ball1_scatter = ax.scatter([], [], [], s=40)

    # Trace lines (optional, short tail)
    tail_len = 80  # number of samples in tail (after stride)
    hand_line, = ax.plot([], [], [], linewidth=1, alpha=0.7)
    b0_line, = ax.plot([], [], [], linewidth=1, alpha=0.7)
    b1_line, = ax.plot([], [], [], linewidth=1, alpha=0.7)

    def init():
        def init():
            hand_scatter._offsets3d = ([], [], [])
            ball0_scatter._offsets3d = ([], [], [])
            ball1_scatter._offsets3d = ([], [], [])

            empty = np.array([])
            hand_line.set_data_3d(empty, empty, empty)
            b0_line.set_data_3d(empty, empty, empty)
            b1_line.set_data_3d(empty, empty, empty)
            return hand_scatter, ball0_scatter, ball1_scatter, hand_line, b0_line, b1_line

    def update(frame_i):
        idx = frame_idxs[frame_i]
        tq = float(t[idx])

        hp = hand_p[idx]
        b0 = ball_position_at_time(0, tq, events_by_ball, t, hand_p, g)
        b1 = ball_position_at_time(1, tq, events_by_ball, t, hand_p, g)

        hand_scatter._offsets3d = ([hp[0]], [hp[1]], [hp[2]])
        ball0_scatter._offsets3d = ([b0[0]], [b0[1]], [b0[2]])
        ball1_scatter._offsets3d = ([b1[0]], [b1[1]], [b1[2]])

        # Tails
        i0 = max(0, idx - tail_len * stride)
        htail = hand_p[i0:idx + 1:stride]
        b0tail = np.array([ball_position_at_time(0, float(tt), events_by_ball, t, hand_p, g)
                           for tt in t[i0:idx + 1:stride]])
        b1tail = np.array([ball_position_at_time(1, float(tt), events_by_ball, t, hand_p, g)
                           for tt in t[i0:idx + 1:stride]])

        hand_line.set_data_3d(htail[:, 0], htail[:, 1], htail[:, 2])
        b0_line.set_data_3d(b0tail[:, 0], b0tail[:, 1], b0tail[:, 2])
        b1_line.set_data_3d(b1tail[:, 0], b1tail[:, 1], b1tail[:, 2])

        ax.set_title(f"2-Ball / 1-Hand Juggling  |  t = {tq:.2f} s")
        return hand_scatter, ball0_scatter, ball1_scatter, hand_line, b0_line, b1_line

    anim = FuncAnimation(fig, update, frames=len(frame_idxs), init_func=init, blit=False, interval=30)

    if save_mp4 is not None:
        try:
            anim.save(save_mp4, dpi=150)
            print(f"Saved animation: {save_mp4}")
        except Exception as e:
            print("Failed to save MP4 (do you have ffmpeg installed and on PATH?)")
            print(f"Error: {e}")

    plt.show()

def retime_events_to_new_waypoint_schedule(
    events: List[Event],
    waypoints: List[Waypoint],
    durations: np.ndarray,
) -> List[Event]:
    """
    Return a NEW list of events whose times (and p/v) are aligned with the NEW waypoint schedule
    defined by durations (waypoint 0 at t=0, waypoint i at sum(durations[:i])).

    Assumes events and waypoints correspond 1:1 and are in the same time order (true for this script).
    """
    ev_sorted = sorted(events, key=lambda e: e.t)
    # Waypoints are already ordered; use their order directly
    wp_new_t = np.cumsum([0.0] + list(durations))

    if len(ev_sorted) != len(waypoints):
        raise ValueError(f"Expected len(events)==len(waypoints), got {len(ev_sorted)} vs {len(waypoints)}")

    events_new: List[Event] = []
    for i, (e_old, w) in enumerate(zip(ev_sorted, waypoints)):
        e = Event(kind=e_old.kind, ball_id=e_old.ball_id, t=float(wp_new_t[i]), p=w.p.copy(), v=w.v.copy())
        events_new.append(e)

    return events_new

def write_pose_cmd_csv(pose_path: str, traj: np.ndarray,
                       roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0):
    """
    Export pose commands matching format:
      t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg
    Uses traj columns: t, x, y, z (meters) -> mm.
    """
    import csv

    with open(pose_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"])
        for row in traj:
            t = float(row[0])
            x_mm = float(row[1]) * 1000.0
            y_mm = float(row[2]) * 1000.0
            z_mm = float(row[3]) * 1000.0
            w.writerow([f"{t:.6f}", f"{x_mm:.6f}", f"{y_mm:.6f}", f"{z_mm:.6f}",
                        f"{roll_deg:.6f}", f"{pitch_deg:.6f}", f"{yaw_deg:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beats", type=int, default=30, help="Number of beats to generate (>=3).")
    ap.add_argument("--beat-period", type=float, default=0.45, help="Beat period Δ [s].")
    ap.add_argument("--dwell", type=float, default=0.08, help="Dwell/hold time after catch before release [s].")
    ap.add_argument("--z-catch", type=float, default=0.15, help="Catch/release Z height [m].")
    ap.add_argument("--height", type=float, default=0.25, help="Desired apex height above z_catch (diagnostic only) [m].")

    ap.add_argument("--r-catch", type=float, default=0.0, help="Catch orbit radius in XY [m]. Start with 0.")
    ap.add_argument("--dphi-deg", type=float, default=0.0, help="Orbit angle increment per beat [deg].")
    ap.add_argument("--phi0-deg", type=float, default=0.0, help="Orbit starting phase [deg].")

    ap.add_argument("--cyl-radius", type=float, default=0.25, help="Working envelope cylinder radius [m].")
    ap.add_argument("--cyl-height", type=float, default=0.50,
                    help="Working envelope cylinder height [m]. z=0 is center.")
    ap.add_argument("--env-eps", type=float, default=0.005, help="Envelope margin inside cylinder [m].")

    ap.add_argument("--opt-time", action="store_true",
                    help="Optimize segment durations to satisfy cylinder constraints.")
    ap.add_argument("--opt-samples-per-seg", type=int, default=5)
    ap.add_argument("--opt-max-scale", type=float, default=4.0)

    ap.add_argument("--sample-hz", type=float, default=200.0, help="Trajectory sample rate [Hz].")
    ap.add_argument("--g", type=float, default=9.81, help="Gravity magnitude [m/s^2].")

    ap.add_argument("--no-catch-vel-match", action="store_true",
                    help="If set, catch waypoint velocity is forced to 0 instead of matching incoming ball velocity.")
    ap.add_argument("--events-out", type=str, default="events.csv", help="Events CSV output path.")
    ap.add_argument("--traj-out", type=str, default="traj.csv", help="Trajectory CSV output path.")

    ap.add_argument("--clamp-traj", action="store_true",
                    help="Clamp the sampled trajectory into the cylinder (hard constraint, may reduce smoothness).")

    ap.add_argument("--start-bottom", action="store_true",
                    help="Start profile at bottom of stroke (z = z_min + env_eps) so first throw can be made immediately.")

    ap.add_argument("--catch-vel-frac", type=float, default=0.75,
                    help="Fraction of incoming ball velocity to match at catch (0.0–1.0).")

    ap.add_argument("--pose-out", type=str, default="pose_cmd.csv",
                    help="Pose command CSV output (t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg).")

    #Plotting arguments
    ap.add_argument("--plot", action="store_true", help="Show stacked plots for position/velocity/accel/jerk.")
    ap.add_argument("--plot-out", type=str, default=None,
                    help="Optional: save stacked plot image (e.g., kinematics.png).")

    ap.add_argument("--animate", action="store_true", help="Show 3D animation of hand + balls.")
    ap.add_argument("--anim-out", type=str, default=None, help="Optional: save animation to MP4 (requires ffmpeg).")

    ap.add_argument("--ws-radius", type=float, default=0.25, help="Workspace cylinder radius for animation [m].")
    ap.add_argument("--anim-stride", type=int, default=2, help="Frame decimation factor for animation speed.")

    args = ap.parse_args()

    z_min = -0.5 * args.cyl_height
    z_max = +0.5 * args.cyl_height
    z_start = (z_min + args.env_eps) if args.start_bottom else args.z_catch

    events, waypoints, stats = generate_events_and_waypoints(
        beats=args.beats,
        beat_period=args.beat_period,
        dwell=args.dwell,
        z_catch=z_start,  # bottom of stroke for clean start with ball in hand
        height=args.height,
        r_catch=args.r_catch,
        dphi_deg=args.dphi_deg,
        phi0_deg=args.phi0_deg,
        cyl_radius=args.cyl_radius,
        z_min=z_min,
        z_max=z_max,
        g_mag=args.g,
        match_catch_velocity=(not args.no_catch_vel_match),
        enforce_waypoints_in_cylinder=True,
    )

    # Force start at rest (safe for real robot tests)
    waypoints[0].v[:] = 0.0
    waypoints[0].p[0] = 0.0
    waypoints[0].p[1] = 0.0

    z_min = -0.5 * args.cyl_height
    z_max = +0.5 * args.cyl_height

    durations0 = np.array([waypoints[i + 1].t - waypoints[i].t for i in range(len(waypoints) - 1)], dtype=float)

    if args.opt_time:
        wps_opt, durations_opt, opt_info = optimize_hand_with_envelope_and_throw_constraints(
            base_waypoints=waypoints,
            events=events,
            durations0=durations0,
            cyl_radius=args.cyl_radius,
            z_min=z_min,
            z_max=z_max,
            g_mag=args.g,
            eps=args.env_eps,
            beta_catch=args.catch_vel_frac,
            catch_soft_weight=5.0,
            time_weight=0.05,
            dev_weight=1.0,
            sample_per_seg=10,
            max_scale=args.opt_max_scale,
            release_v_tol_frac=0.01,
            use_release_tol_band=False,  # release is hard by construction
        )

        print("Optimization:")
        print(f"  success: {opt_info['success']}")
        print(f"  message: {opt_info['message']}")
        print(f"  total_time nominal: {opt_info['total_time_nominal']:.3f} -> opt: {opt_info['total_time_opt']:.3f}")
        print(f"  worst constraint margin (>=0 ok): {opt_info['worst_margin']:.6f}")

        traj = build_quintic_trajectory_from_durations(wps_opt, durations_opt, sample_hz=args.sample_hz)

        # ✅ Retime events onto the optimized waypoint timeline for animation/ballistics
        events_for_anim = retime_events_to_new_waypoint_schedule(events, wps_opt, durations_opt)
    else:
        traj = build_quintic_trajectory(waypoints, sample_hz=args.sample_hz)
        events_for_anim = events

    # Check violations
    p = traj[:, 1:4]
    r = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    viol = np.any(r > args.cyl_radius + 1e-9) or np.any(p[:, 2] < z_min - 1e-9) or np.any(p[:, 2] > z_max + 1e-9)

    if viol:
        print("WARNING: Trajectory leaves cylinder between waypoints.")
        if args.clamp_traj:
            traj = clamp_traj_to_cylinder(traj, args.cyl_radius, z_min, z_max)
            print("Clamped trajectory into cylinder and recomputed v/a.")
        else:
            print("Tip: re-run with --clamp-traj or increase beat-period / reduce height / disable catch vel match.")

    write_events_csv(args.events_out, events)
    write_traj_csv(args.traj_out, traj)
    write_pose_cmd_csv(args.pose_out, traj, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)
    print(f"Wrote: {args.pose_out}")

    # Print quick diagnostics
    v_max = np.max(np.linalg.norm(traj[:, 4:7], axis=1))
    a_max = np.max(np.linalg.norm(traj[:, 7:10], axis=1))

    print("Generated 2-ball / 1-hand juggling pattern")
    print(f"  beats: {args.beats}")
    print(f"  beat_period Δ: {stats['Δ']:.3f} s")
    print(f"  dwell: {stats['dwell']:.3f} s")
    print(f"  flight time T_flight: {stats['T_flight']:.3f} s")
    print(f"  match_catch_velocity: {stats['match_catch_velocity']}")
    print(f"  catch orbit radius r_catch: {args.r_catch:.3f} m, dphi: {args.dphi_deg:.1f} deg/beat")
    print(f"  implied apex z: min={stats['apex_z_min']:.3f}, mean={stats['apex_z_mean']:.3f}, max={stats['apex_z_max']:.3f} (desired {stats['desired_apex_z']:.3f})")
    print(f"  traj samples: {traj.shape[0]} @ {args.sample_hz:.1f} Hz")
    print(f"  max speed: {v_max:.3f} m/s, max accel: {a_max:.3f} m/s^2")
    print(f"Wrote: {args.events_out}, {args.traj_out}")

    #Plotting
    if args.plot:
        plot_stacked_kinematics(traj, show=True, save_path=args.plot_out)

    if args.animate:
        animate_juggling_3d(
            traj=traj,
            events=events_for_anim,
            workspace_radius=args.cyl_radius,
            z_min=z_min,
            z_max=z_max,
            g_mag=args.g,
            stride=max(1, args.anim_stride),
            save_mp4=args.anim_out,
        )



if __name__ == "__main__":
    main()
