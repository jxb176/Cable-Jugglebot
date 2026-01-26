# jugglepath.py
# ------------------------------------------------------------
# Waypoint + Segment based path authoring (N waypoints, N-1 segments)
#
# Segment time laws:
#   - "linear": constant speed along line
#   - "s_curve_monotonic": existing LineDVNoCoastScaled (dv-only, monotonic accel/decel)
#   - "s_curve": NEW generic jerk/accel-limited S-curve with optional cruise (7-seg structure)
#
# s_curve implementation is adapted from your playground example:
#   path_playground_gui_scurve_basic_3d.py
#   - min-time dv phases + optional cruise
#   - exact constant-jerk integration sampling
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal, Tuple
import math
import numpy as np


# ----------------------------
# Core containers
# ----------------------------

@dataclass
class State3D:
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=float).reshape(3)
        self.v = np.asarray(self.v, dtype=float).reshape(3)
        self.a = np.asarray(self.a, dtype=float).reshape(3)


@dataclass
class SegmentResult:
    traj: np.ndarray  # (N,13): t,x,y,z,vx,vy,vz,ax,ay,az,jx,jy,jz
    end_state: State3D
    info: Dict[str, object]


@dataclass
class PathResult:
    traj: np.ndarray
    segment_infos: List[Dict[str, object]]
    end_state: State3D


# ----------------------------
# Waypoints / segments
# ----------------------------

@dataclass
class Waypoint:
    p: np.ndarray
    v: Optional[np.ndarray] = None
    a: Optional[np.ndarray] = None
    t: Optional[float] = None

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=float).reshape(3)
        if self.v is not None:
            self.v = np.asarray(self.v, dtype=float).reshape(3)
        if self.a is not None:
            self.a = np.asarray(self.a, dtype=float).reshape(3)
        if self.t is not None:
            self.t = float(self.t)


@dataclass
class SegmentSpec:
    curve: Literal["line"] = "line"
    time_law: Literal["linear", "s_curve_monotonic", "s_curve", "wait"] = "linear"

    # Optional: fix duration. Used by "linear" now; "s_curve" will ignore for the moment.
    duration: Optional[float] = None

    # For s_curve and s_curve_monotonic:
    # These are ALONG-PATH scalar limits (units of m/s^2 and m/s^3 in s-space).
    # For line segments, this is exactly physical along-path accel/jerk.
    accel_ref: float = 1.0
    jerk_ref: float = 10.0

    # Optional along-path speed cap for s_curve (enables "cruise at vmax")
    v_max: Optional[float] = None


# ----------------------------
# Primitive base
# ----------------------------

class Primitive3D:
    def generate(self, start: State3D, sample_hz: float) -> SegmentResult:
        raise NotImplementedError


# ============================================================
# 1D constant-jerk helpers (used by both monotonic + s_curve)
# ============================================================

def _integrate_const_jerk_1d(s: float, v: float, a: float, j: float, dt: float) -> Tuple[float, float, float]:
    s1 = s + v * dt + 0.5 * a * dt * dt + (1.0 / 6.0) * j * dt ** 3
    v1 = v + a * dt + 0.5 * j * dt * dt
    a1 = a + j * dt
    return s1, v1, a1


def _build_min_time_dv_segments(v_start: float, v_end: float, amax: float, jmax: float) -> List[Tuple[float, float]]:
    """
    Min-time jerk-limited change in velocity from v_start to v_end,
    starting and ending at a=0. Returns list of (jerk, duration).
    Matches the pattern in your GUI example. :contentReference[oaicite:4]{index=4}
    """
    dv = v_end - v_start
    if abs(dv) < 1e-15:
        return []

    amax = max(1e-15, float(amax))
    jmax = max(1e-15, float(jmax))

    sgn = 1.0 if dv > 0 else -1.0
    dv_abs = abs(dv)

    a_peak = math.sqrt(dv_abs * jmax)
    if a_peak <= amax + 1e-15:
        Tj = a_peak / jmax
        return [(sgn * jmax, Tj), (-sgn * jmax, Tj)]

    Tj = amax / jmax
    Ta = dv_abs / amax - Tj
    if Ta < 0:
        Ta = 0.0

    segs: List[Tuple[float, float]] = [(sgn * jmax, Tj)]
    if Ta > 0:
        segs.append((0.0, Ta))
    segs.append((-sgn * jmax, Tj))
    return segs


def _phase_distance_time(v_start: float, v_end: float, amax: float, jmax: float) -> Tuple[
    float, float, List[Tuple[float, float]]]:
    """
    Returns (distance, time, segments) for the min-time dv phase.
    Mirrors the approach in your GUI example. :contentReference[oaicite:5]{index=5}
    """
    segs = _build_min_time_dv_segments(v_start, v_end, amax, jmax)
    s = 0.0
    v = float(v_start)
    a = 0.0
    t = 0.0
    for (j, dur) in segs:
        s, v, a = _integrate_const_jerk_1d(s, v, a, j, dur)
        t += dur
    return s, t, segs


def _simulate_segments_1d(
        segments: List[Tuple[float, float]],
        s0: float,
        v0: float,
        a0: float,
        dt: float
) -> np.ndarray:
    """
    Sample [t, s, v, a, j] by integrating piecewise-constant jerk.
    Matches your GUI example. :contentReference[oaicite:6]{index=6}
    """
    rows = []
    t = 0.0
    s = float(s0)
    v = float(v0)
    a = float(a0)

    for (j, dur) in segments:
        if dur <= 0:
            continue
        t_end = t + dur
        while t < t_end - 1e-12:
            dt_step = min(dt, t_end - t)
            s, v, a = _integrate_const_jerk_1d(s, v, a, j, dt_step)
            t += dt_step
            rows.append([t, s, v, a, j])

    return np.array(rows, dtype=float) if rows else np.zeros((0, 5), dtype=float)


# ============================================================
# Time laws / primitives
# ============================================================


class Wait(Primitive3D):
    """Hold position for a fixed duration. Outputs v=a=j=0 throughout."""

    def __init__(self, duration: float):
        self.duration = float(duration)
        if self.duration <= 0:
            raise ValueError("Wait duration must be > 0")

    def generate(self, start: State3D, sample_hz: float) -> SegmentResult:
        dt = 1.0 / float(sample_hz)
        n = max(2, int(np.ceil(self.duration / dt)) + 1)
        t = np.linspace(0.0, self.duration, n, dtype=float)

        p = start.p.reshape(1, 3).astype(float)
        P = np.repeat(p, n, axis=0)
        Z = np.zeros((n, 3), dtype=float)

        traj = np.zeros((n, 13), dtype=float)
        traj[:, 0] = t
        traj[:, 1:4] = P
        traj[:, 4:7] = Z
        traj[:, 7:10] = Z
        traj[:, 10:13] = Z

        end_state = State3D(p=start.p.copy(), v=np.zeros(3), a=np.zeros(3))
        info = {"primitive": "wait", "duration": float(self.duration)}
        return SegmentResult(traj=traj, end_state=end_state, info=info)


class LineLinear(Primitive3D):
    """Constant speed along the line. Uses duration if provided, else nominal_speed."""

    def __init__(self, p1: np.ndarray, duration: Optional[float] = None, nominal_speed: float = 0.5):
        self.p1 = np.asarray(p1, dtype=float).reshape(3)
        self.duration = None if duration is None else float(duration)
        self.nominal_speed = float(nominal_speed)

    def generate(self, start: State3D, sample_hz: float) -> SegmentResult:
        dt = 1.0 / float(sample_hz)
        p0 = start.p
        d = self.p1 - p0
        L = float(np.linalg.norm(d))

        if L < 1e-12:
            traj = np.zeros((1, 13), dtype=float)
            traj[0, 0] = 0.0
            traj[0, 1:4] = p0
            traj[0, 4:7] = start.v
            traj[0, 7:10] = start.a
            info = {"mode": "line_linear", "degenerate": True, "L": 0.0, "t_total": 0.0}
            return SegmentResult(traj=traj, end_state=start, info=info)

        u = d / L
        if self.duration is not None:
            T = max(1e-9, self.duration)
            vmag = L / T
        else:
            vmag = max(1e-9, abs(self.nominal_speed))
            T = L / vmag

        n = max(2, int(math.ceil(T / dt)) + 1)
        t = np.linspace(0.0, T, n)
        s = (L / T) * t

        p = p0[None, :] + s[:, None] * u[None, :]
        v = np.full((n, 1), L / T) * u[None, :]
        a = np.zeros((n, 3), dtype=float)
        j = np.zeros((n, 3), dtype=float)

        traj = np.zeros((n, 13), dtype=float)
        traj[:, 0] = t
        traj[:, 1:4] = p
        traj[:, 4:7] = v
        traj[:, 7:10] = a
        traj[:, 10:13] = j

        end = State3D(p=traj[-1, 1:4].copy(), v=traj[-1, 4:7].copy(), a=traj[-1, 7:10].copy())
        info = {"mode": "line_linear", "L": L, "t_total": float(traj[-1, 0])}
        return SegmentResult(traj=traj, end_state=end, info=info)


class LineDVNoCoastScaled(Primitive3D):
    """
    Existing monotonic accel/decel time law (dv-only, scaled to hit distance).
    Kept as-is; this is your s_curve_monotonic.

    NOTE: This class already exists in your current jugglepath.py. :contentReference[oaicite:7]{index=7}
    The body below is exactly what you uploaded, left unchanged.
    """

    def __init__(
            self,
            p1: np.ndarray,
            v1_along: float,
            accel_ref: float,
            jerk_ref: float,
            scale_accel: bool = True,
            scale_jerk: bool = True,
            k_min: float = 1e-4,
            k_max: float = 1e4,
            grid_points: int = 81,
            refine_bisect_iters: int = 40,
    ):
        self.p1 = np.asarray(p1, dtype=float).reshape(3)
        self.v1_along = float(v1_along)
        self.accel_ref = float(accel_ref)
        self.jerk_ref = float(jerk_ref)
        self.scale_accel = bool(scale_accel)
        self.scale_jerk = bool(scale_jerk)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.grid_points = int(grid_points)
        self.refine_bisect_iters = int(refine_bisect_iters)

    def _effective_refs(self, k: float) -> Tuple[float, float]:
        a_used = self.accel_ref * k if self.scale_accel else self.accel_ref
        j_used = self.jerk_ref * k if self.scale_jerk else self.jerk_ref
        return float(a_used), float(j_used)

    def _choose_k_best_match(self, vs0: float, L: float) -> Tuple[float, float]:
        L = abs(float(L))

        if not self.scale_accel and not self.scale_jerk:
            a_used, j_used = self._effective_refs(1.0)
            d_base, _, _ = _phase_distance_time(vs0, self.v1_along, a_used, j_used)
            return 1.0, float(d_base)

        k_min = max(1e-12, self.k_min)
        k_max = max(k_min * 1.0001, self.k_max)
        n = max(11, self.grid_points)

        ks = np.logspace(math.log10(k_min), math.log10(k_max), n)

        best_k = float(ks[0])
        best_err = float("inf")
        best_d = 0.0

        for k in ks:
            a_used, j_used = self._effective_refs(float(k))
            d_base, _, _ = _phase_distance_time(vs0, self.v1_along, a_used, j_used)
            err = abs(abs(d_base) - L)
            if err < best_err:
                best_err = err
                best_k = float(k)
                best_d = float(d_base)

        def f(k: float) -> float:
            a_used, j_used = self._effective_refs(float(k))
            d_base, _, _ = _phase_distance_time(vs0, self.v1_along, a_used, j_used)
            return abs(d_base) - L

        idx = int(np.argmin(np.abs(ks - best_k)))
        lo = float(ks[max(0, idx - 1)])
        hi = float(ks[min(len(ks) - 1, idx + 1)])
        f_lo = f(lo)
        f_hi = f(hi)

        k_used = best_k
        if f_lo * f_hi < 0.0:
            a = lo
            b = hi
            fa = f_lo
            fb = f_hi
            for _ in range(max(1, self.refine_bisect_iters)):
                m = 0.5 * (a + b)
                fm = f(m)
                if fa * fm <= 0.0:
                    b = m
                    fb = fm
                else:
                    a = m
                    fa = fm
            k_used = 0.5 * (a + b)

        a_used, j_used = self._effective_refs(k_used)
        d_base, _, _ = _phase_distance_time(vs0, self.v1_along, a_used, j_used)
        return float(k_used), float(d_base)

    def generate(self, start: State3D, sample_hz: float) -> SegmentResult:
        dt = 1.0 / float(sample_hz)

        p0 = start.p.astype(float)
        p1 = self.p1.astype(float)
        d = p1 - p0
        L = float(np.linalg.norm(d))

        if L < 1e-12:
            traj = np.zeros((1, 13), dtype=float)
            traj[0, 0] = 0.0
            traj[0, 1:4] = p0
            traj[0, 4:7] = start.v
            traj[0, 7:10] = start.a
            info = {"mode": "line_dv_no_coast_scaled", "degenerate": 1.0, "L": 0.0}
            end = State3D(p=p0.copy(), v=start.v.copy(), a=start.a.copy())
            return SegmentResult(traj=traj, end_state=end, info=info)

        u = d / L

        v0_parallel = float(np.dot(start.v, u))
        lateral_speed_in = float(np.linalg.norm(start.v - v0_parallel * u))
        vs0 = float(v0_parallel)

        k_used, d_base = self._choose_k_best_match(vs0=vs0, L=L)
        a_used, j_used = self._effective_refs(k_used)

        _, t_base, segs = _phase_distance_time(vs0, self.v1_along, a_used, j_used)
        samples = _simulate_segments_1d(segs, s0=0.0, v0=vs0, a0=0.0, dt=dt)

        if samples.shape[0] == 0:
            traj = np.zeros((1, 13), dtype=float)
            traj[0, 0] = 0.0
            traj[0, 1:4] = p0
            traj[0, 4:7] = v0_parallel * u
            traj[0, 7:10] = np.zeros(3)
            info = {
                "mode": "line_dv_no_coast_scaled",
                "degenerate": 1.0,
                "L": L,
                "k_used": k_used,
                "accel_ref_used": a_used,
                "jerk_ref_used": j_used,
                "d_base": d_base,
                "L_error_abs": abs(abs(d_base) - L),
                "lateral_speed_in": lateral_speed_in,
                "t_total": 0.0,
            }
            end = State3D(p=p0.copy(), v=v0_parallel * u, a=np.zeros(3))
            return SegmentResult(traj=traj, end_state=end, info=info)

        # force end to exactly p1 by shifting s

        s_end = float(samples[-1, 1])
        if abs(s_end) < 1e-15:
            raise ValueError("Profile generation failed (near-zero distance)")
        scale = float(L / s_end)
        samples[:, 1] *= scale  # s
        samples[:, 2] *= scale  # sdot
        samples[:, 3] *= scale  # sddot
        samples[:, 4] *= scale  # sjerk
        samples[-1, 1] = float(L)
        t = samples[:, 0]
        s = samples[:, 1]
        vs = samples[:, 2]
        a_s = samples[:, 3]
        j_s = samples[:, 4]

        p = p0[None, :] + s[:, None] * u[None, :]
        v = vs[:, None] * u[None, :]
        a = a_s[:, None] * u[None, :]
        j = j_s[:, None] * u[None, :]

        traj = np.zeros((samples.shape[0], 13), dtype=float)
        traj[:, 0] = t
        traj[:, 1:4] = p
        traj[:, 4:7] = v
        traj[:, 7:10] = a
        traj[:, 10:13] = j

        end = State3D(p=traj[-1, 1:4].copy(), v=traj[-1, 4:7].copy(), a=traj[-1, 7:10].copy())

        info = {
            "mode": "line_dv_no_coast_scaled",
            "L": float(L),
            "k_used": float(k_used),
            "scale_accel": float(self.scale_accel),
            "scale_jerk": float(self.scale_jerk),
            "accel_ref_used": float(a_used),
            "jerk_ref_used": float(j_used),
            "t_total": float(traj[-1, 0]),
            "t_base": float(t_base),
            "d_base": float(d_base),
            "L_error_abs": float(abs(abs(d_base) - L)),
            "a_peak": float(np.max(np.linalg.norm(traj[:, 7:10], axis=1))),
            "j_peak": float(np.max(np.linalg.norm(traj[:, 10:13], axis=1))),
            "vs0": float(vs0),
            "vs1_cmd": float(self.v1_along),
            "lateral_speed_in": float(lateral_speed_in),
        }

        return SegmentResult(traj=traj, end_state=end, info=info)


class LineSCurve(Primitive3D):
    """
    Generic S-curve time law (jerk + accel limited) along a LINE, with optional cruise.
    This is the "s_curve" you asked for.

    This follows the same logic as generate_scurve_normalized() in your playground script:
      - choose v_peak (possibly v_max) and optional cruise to match distance
      - build segments = dv_up + (optional cruise) + dv_down
      - simulate under constant jerk sampling
    :contentReference[oaicite:8]{index=8}

    Boundary conditions supported:
      - along-path v0, v1 (sdot0, sdot1)
      - assumes a0=a1=0 for now (classic 7-seg S-curve)
    """

    def __init__(
            self,
            p1: np.ndarray,
            v1_along: float,
            amax: float,
            jmax: float,
            v_max: Optional[float] = None,
    ):
        self.p1 = np.asarray(p1, dtype=float).reshape(3)
        self.v1_along = float(v1_along)
        self.amax = float(amax)
        self.jmax = float(jmax)
        self.v_max = None if v_max is None else float(v_max)

    def _generate_1d(self, v0: float, v1: float, L: float, sample_hz: float) -> Tuple[np.ndarray, Dict[str, object]]:
        if sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        if self.amax <= 0 or self.jmax <= 0:
            raise ValueError("amax/jmax must be > 0")
        if L <= 0:
            raise ValueError("L must be > 0")

        amax = self.amax
        jmax = self.jmax

        # Distance required for a given peak speed (min-time accel to vp, optional cruise, min-time decel to v1)
        def d_min_for_peak(vp: float) -> float:
            d1, _, _ = _phase_distance_time(v0, vp, amax, jmax)
            d2, _, _ = _phase_distance_time(vp, v1, amax, jmax)
            return float(d1 + d2)

        # Choose a feasible v_peak:
        mode = ""
        t_cruise = 0.0

        if self.v_max is not None:
            v_hi = max(0.0, float(self.v_max))
        else:
            # no explicit vmax: expand upper bound until d_min_for_peak(v_hi) >= L
            v_hi = max(0.0, v0, v1, 1e-3)
            for _ in range(80):
                if d_min_for_peak(v_hi) >= L - 1e-12:
                    break
                v_hi *= 2.0
            # if still not enough, it's basically infeasible with given a/j (should be rare)
            # but guard anyway:
            if d_min_for_peak(v_hi) < L - 1e-9:
                raise ValueError("Unable to bracket v_peak for distance match (check limits).")

        d_at_hi = d_min_for_peak(v_hi)

        if self.v_max is not None and d_at_hi <= L + 1e-12:
            # We can reach v_max and still need more distance -> cruise at v_max
            v_peak = v_hi
            d_min = d_at_hi
            t_cruise = (L - d_min) / v_peak if v_peak > 1e-15 else 0.0
            mode = "cruise_at_vmax" if t_cruise > 0 else "no_cruise_at_vmax"
        else:
            # Need a smaller v_peak so that d_min_for_peak(v_peak) == L (no cruise)
            v_lo = max(0.0, v0, v1)
            if d_min_for_peak(v_lo) > L + 1e-12:
                raise ValueError("infeasible: distance too short for given endpoint velocities and limits")

            lo, hi = v_lo, v_hi
            for _ in range(90):
                mid = 0.5 * (lo + hi)
                if d_min_for_peak(mid) > L:
                    hi = mid
                else:
                    lo = mid
                if abs(d_min_for_peak(mid) - L) < 1e-12:
                    break
            v_peak = lo
            t_cruise = 0.0
            mode = "limited_peak_no_cruise"

        segs_up = _build_min_time_dv_segments(v0, v_peak, amax, jmax)
        segs_dn = _build_min_time_dv_segments(v_peak, v1, amax, jmax)

        segments: List[Tuple[float, float]] = []
        segments.extend(segs_up)
        if t_cruise > 0:
            segments.append((0.0, t_cruise))  # jerk=0, accel constant (0), velocity constant
        segments.extend(segs_dn)

        dt = 1.0 / float(sample_hz)
        samp = _simulate_segments_1d(segments, s0=0.0, v0=v0, a0=0.0, dt=dt)
        if samp.shape[0] == 0:
            raise ValueError("degenerate profile")

        # Make the sampled distance land exactly at L without changing the start point.
        # IMPORTANT: Do NOT "shift" the whole s(t) curve (that moves the start away from 0 and
        # causes visible overshoot/undershoot in 3D). Instead, scale s and its derivatives
        # consistently (same approach you used in the Z-only playground).
        s_end = float(samp[-1, 1])
        if abs(s_end) < 1e-15:
            raise ValueError("Profile generation failed (near-zero distance)")
        scale = float(L / s_end)

        samp[:, 1] *= scale  # s
        samp[:, 2] *= scale  # sdot
        samp[:, 3] *= scale  # sddot
        samp[:, 4] *= scale  # sjerk

        # Tiny numerical cleanup
        samp[-1, 1] = float(L)

        info = {
            "mode": mode,
            "v_peak": float(v_peak),
            "t_total": float(samp[-1, 0]),
            "t_cruise": float(t_cruise),
            "L": float(L),
        }
        return samp, info

    def generate(self, start: State3D, sample_hz: float) -> SegmentResult:
        p0 = start.p.astype(float)
        p1 = self.p1.astype(float)
        d = p1 - p0
        L = float(np.linalg.norm(d))

        if L < 1e-12:
            traj = np.zeros((1, 13), dtype=float)
            traj[0, 0] = 0.0
            traj[0, 1:4] = p0
            traj[0, 4:7] = start.v
            traj[0, 7:10] = start.a
            info = {"mode": "line_s_curve", "degenerate": True, "L": 0.0}
            end = State3D(p=p0.copy(), v=start.v.copy(), a=start.a.copy())
            return SegmentResult(traj=traj, end_state=end, info=info)

        u = d / L
        v0_along = float(np.dot(start.v, u))
        v1_along = float(self.v1_along)

        samp, info1d = self._generate_1d(v0=v0_along, v1=v1_along, L=L, sample_hz=sample_hz)

        t = samp[:, 0]
        s = samp[:, 1]
        vs = samp[:, 2]
        a_s = samp[:, 3]
        j_s = samp[:, 4]

        p = p0[None, :] + s[:, None] * u[None, :]
        v = vs[:, None] * u[None, :]
        a = a_s[:, None] * u[None, :]
        j = j_s[:, None] * u[None, :]

        traj = np.zeros((samp.shape[0], 13), dtype=float)
        traj[:, 0] = t
        traj[:, 1:4] = p
        traj[:, 4:7] = v
        traj[:, 7:10] = a
        traj[:, 10:13] = j

        end = State3D(p=traj[-1, 1:4].copy(), v=traj[-1, 4:7].copy(), a=traj[-1, 7:10].copy())

        info = {
            "mode": "line_s_curve",
            **info1d,
            "a_peak": float(np.max(np.abs(a_s))) if a_s.size else 0.0,
            "j_peak": float(np.max(np.abs(j_s))) if j_s.size else 0.0,
            "v0_along": float(v0_along),
            "v1_along_cmd": float(v1_along),
        }
        return SegmentResult(traj=traj, end_state=end, info=info)


# ============================================================
# JugglePath: Waypoints + Segments container
# ============================================================

class JugglePath:
    """
    Authoritative model:
      - waypoints: N
      - segments : N-1

    Authoring workflow:
      path = JugglePath(sample_hz=..., start=State3D(...optional...))
      path.add_segment(p=..., v=..., a=..., t=..., time_law="s_curve", accel_ref=..., jerk_ref=..., v_max=...)
      res = path.build()

    Default semantics:
      - if waypoint v/a not provided -> treated as 0 for now
      - s_curve_monotonic: accel endpoints default 0 unless waypoint provides (but current primitive doesn't enforce accel BCs)
      - s_curve: assumes endpoint accel=0 (classic 7-seg S-curve). If waypoint accel is nonzero we record a warning.
    """

    def __init__(self, sample_hz: float, start: Optional[State3D] = None):
        self.sample_hz = float(sample_hz)

        if start is None:
            start = State3D(p=np.zeros(3), v=np.zeros(3), a=np.zeros(3))

        self.waypoints: List[Waypoint] = [Waypoint(p=start.p.copy(), v=start.v.copy(), a=start.a.copy(), t=0.0)]
        self.segments: List[SegmentSpec] = []

    def add_segment(
            self,
            p: np.ndarray,
            v: Optional[np.ndarray] = None,
            a: Optional[np.ndarray] = None,
            t: Optional[float] = None,
            *,
            curve: Literal["line"] = "line",
            time_law: Literal["linear", "s_curve_monotonic", "s_curve", "wait"] = "linear",
            duration: Optional[float] = None,
            accel_ref: float = 1.0,
            jerk_ref: float = 10.0,
            v_max: Optional[float] = None,
    ) -> "JugglePath":
        self.segments.append(SegmentSpec(
            curve=curve,
            time_law=time_law,
            duration=duration,
            accel_ref=float(accel_ref),
            jerk_ref=float(jerk_ref),
            v_max=(None if v_max is None else float(v_max)),
        ))
        self.waypoints.append(Waypoint(p=p, v=v, a=a, t=t))
        return self

    def add_wait(self, duration: float) -> "JugglePath":
        """
        Append a WAIT primitive that holds the current pose for `duration` seconds.

        Semantics:
          - Adds a new waypoint at the SAME position as the previous waypoint.
          - Sets waypoint v=a=0 (i.e., the wait represents a stop/hold).
          - Requires duration > 0.

        You can also get a wait by calling add_segment(p=current_p, time_law="wait", duration=...).
        """
        duration = float(duration)
        if duration <= 0:
            raise ValueError("duration must be > 0 for add_wait()")

        p = self.waypoints[-1].p.copy()
        return self.add_segment(
            p=p,
            v=np.zeros(3),
            a=np.zeros(3),
            time_law="wait",
            duration=duration,
        )

    def set_waypoint(self, i: int, *, p=None, v=None, a=None, t=None) -> None:
        wp = self.waypoints[i]
        if p is not None: wp.p = np.asarray(p, dtype=float).reshape(3)
        if v is not None: wp.v = np.asarray(v, dtype=float).reshape(3)
        if a is not None: wp.a = np.asarray(a, dtype=float).reshape(3)
        if t is not None: wp.t = float(t)

    def set_segment(self, i: int, **kwargs: Any) -> None:
        seg = self.segments[i]
        for k, val in kwargs.items():
            if not hasattr(seg, k):
                raise AttributeError(f"SegmentSpec has no field '{k}'")
            setattr(seg, k, val)

    # ---- helpers ----

    def _segment_duration(self, i: int) -> Optional[float]:
        seg = self.segments[i]
        if seg.duration is not None:
            return float(seg.duration)

        w0 = self.waypoints[i]
        w1 = self.waypoints[i + 1]
        if w0.t is not None and w1.t is not None:
            T = float(w1.t - w0.t)
            if T <= 0:
                raise ValueError(f"Non-positive waypoint time for segment {i}: {w0.t} -> {w1.t}")
            return T
        return None

    def _along_path_bc(self, wp: Waypoint, u: np.ndarray) -> Tuple[float, float]:
        # defaults to 0 unless defined (your requested behavior)
        sd = float(np.dot(wp.v, u)) if wp.v is not None else 0.0
        sdd = float(np.dot(wp.a, u)) if wp.a is not None else 0.0
        return sd, sdd

    def _materialize_primitive(self, i: int, start_state: State3D) -> Tuple[Primitive3D, Dict[str, object]]:
        seg = self.segments[i]
        w0 = self.waypoints[i]
        w1 = self.waypoints[i + 1]

        dp = (w1.p - start_state.p).astype(float)
        L = float(np.linalg.norm(dp))
        u = dp / L if L > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=float)

        # boundary conditions along path direction
        sd0, sdd0 = self._along_path_bc(w0, u)
        sd1, sdd1 = self._along_path_bc(w1, u)

        extra_info: Dict[str, object] = {
            "sd0": float(sd0),
            "sdd0": float(sdd0),
            "sd1": float(sd1),
            "sdd1": float(sdd1),
        }

        if seg.time_law == "wait":
            T = self._segment_duration(i)
            if T is None:
                raise ValueError("WAIT segment requires a duration (SegmentSpec.duration or waypoint times).")
            if L > 1e-9:
                raise ValueError(
                    "WAIT segment requires no position change (next waypoint must equal current position).")
            return Wait(duration=T), extra_info

        if seg.time_law == "linear":
            return LineLinear(p1=w1.p, duration=self._segment_duration(i)), extra_info

        if seg.time_law == "s_curve_monotonic":
            # Monotonic dv primitive; accel BCs are part of the contract but not enforced yet.
            # Defaults are already correct because _along_path_bc defaults sdd to 0.
            return LineDVNoCoastScaled(
                p1=w1.p,
                v1_along=sd1,
                accel_ref=seg.accel_ref,
                jerk_ref=seg.jerk_ref,
                scale_accel=True,
                scale_jerk=True,
            ), extra_info

        if seg.time_law == "s_curve":
            # Classic 7-seg assumes a(0)=a(T)=0.
            # If waypoint a is nonzero, record warning (do not silently "pretend").
            warn = ""
            if abs(sdd0) > 1e-9 or abs(sdd1) > 1e-9:
                warn = "WARNING: s_curve currently assumes endpoint along-path acceleration = 0; waypoint accel is ignored."
            if warn:
                extra_info["warning"] = warn

            return LineSCurve(
                p1=w1.p,
                v1_along=sd1,
                amax=seg.accel_ref,
                jmax=seg.jerk_ref,
                v_max=seg.v_max,
            ), extra_info

        raise ValueError(f"Unknown time_law: {seg.time_law}")

    # ---- build ----

    def build(self) -> PathResult:
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")
        if len(self.waypoints) < 2:
            empty = np.zeros((0, 13), dtype=float)
            return PathResult(traj=empty, segment_infos=[], end_state=State3D(np.zeros(3), np.zeros(3), np.zeros(3)))
        if len(self.segments) != len(self.waypoints) - 1:
            raise ValueError("Need exactly N-1 segments for N waypoints.")

        all_traj: List[np.ndarray] = []
        infos: List[Dict[str, object]] = []

        w0 = self.waypoints[0]
        cur = State3D(
            p=w0.p.copy(),
            v=(w0.v.copy() if w0.v is not None else np.zeros(3)),
            a=(w0.a.copy() if w0.a is not None else np.zeros(3)),
        )

        t_offset = 0.0

        for i in range(len(self.segments)):
            prim, extra = self._materialize_primitive(i, cur)
            r = prim.generate(start=cur, sample_hz=self.sample_hz)

            traj = r.traj.copy()
            traj[:, 0] += t_offset

            # De-dup boundary sample
            if all_traj and traj.shape[0] > 0 and abs(traj[0, 0] - t_offset) < 1e-12:
                traj = traj[1:, :]

            if traj.shape[0] > 0:
                t_offset = float(traj[-1, 0])

            all_traj.append(traj)

            infos.append({
                "segment_index": i,
                "curve": self.segments[i].curve,
                "time_law": self.segments[i].time_law,
                **extra,
                **(r.info if isinstance(r.info, dict) else {"info": r.info}),
            })

            cur = r.end_state  # chain

        full = np.vstack(all_traj) if all_traj else np.zeros((0, 13), dtype=float)
        return PathResult(traj=full, segment_infos=infos, end_state=cur)
