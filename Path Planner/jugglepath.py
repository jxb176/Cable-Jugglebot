# juggle_path_xyz.py
# ------------------------------------------------------------
# Standalone 3D trajectory generator built from chained path primitives.
#
# - JugglePath: chains primitives so end state feeds next start state
# - LineDVNoCoastScaled: straight-line primitive with jerk/accel limited dv,
#   no cruise, optional scaling of accel_ref and/or jerk_ref by a single k
#
# This module intentionally contains no executable demo. See examples/.
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import math
import numpy as np


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class State3D:
    """3D kinematic state."""
    p: np.ndarray   # shape (3,)
    v: np.ndarray   # shape (3,)
    a: np.ndarray   # shape (3,)

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=float).reshape(3)
        self.v = np.asarray(self.v, dtype=float).reshape(3)
        self.a = np.asarray(self.a, dtype=float).reshape(3)


@dataclass
class SegmentResult:
    traj: np.ndarray          # (N, 13): [t,x,y,z, vx,vy,vz, ax,ay,az, jx,jy,jz]
    end_state: State3D
    info: Dict[str, float]


@dataclass
class PathResult:
    traj: np.ndarray
    segment_infos: List[Dict[str, float]]
    end_state: State3D


# ----------------------------
# 1D core under constant jerk (path coordinate s in meters)
# ----------------------------

def _integrate_const_jerk_1d(s: float, v: float, a: float, j: float, dt: float) -> Tuple[float, float, float]:
    s1 = s + v * dt + 0.5 * a * dt * dt + (1.0 / 6.0) * j * dt**3
    v1 = v + a * dt + 0.5 * j * dt * dt
    a1 = a + j * dt
    return s1, v1, a1


def _build_min_time_dv_segments(v_start: float, v_end: float, amax: float, jmax: float) -> List[Tuple[float, float]]:
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


def _phase_distance_time(v_start: float, v_end: float, amax: float, jmax: float) -> Tuple[float, float, List[Tuple[float, float]]]:
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


class Primitive3D:
    def generate(self, start: State3D, sample_hz: float) -> SegmentResult:
        raise NotImplementedError


class LineDVNoCoastScaled(Primitive3D):
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

        p0 = start.p
        dp = self.p1 - p0
        L = float(np.linalg.norm(dp))

        if L < 1e-12:
            traj = np.zeros((1, 13), dtype=float)
            traj[0, 0] = 0.0
            traj[0, 1:4] = p0
            traj[0, 4:7] = start.v
            traj[0, 7:10] = start.a
            info = {"mode": "line_dv_no_coast_scaled", "degenerate": 1.0, "L": 0.0, "k_used": 1.0, "d_base": 0.0}
            return SegmentResult(traj=traj, end_state=start, info=info)

        u = dp / L

        vs0 = float(np.dot(start.v, u))
        v0_parallel = u * vs0
        v0_lateral = start.v - v0_parallel
        lateral_speed_in = float(np.linalg.norm(v0_lateral))

        k_used, d_base = self._choose_k_best_match(vs0=vs0, L=L)
        a_used, j_used = self._effective_refs(k_used)

        _, t_base, segs = _phase_distance_time(vs0, self.v1_along, a_used, j_used)
        samples = _simulate_segments_1d(segs, s0=0.0, v0=vs0, a0=0.0, dt=dt)

        if samples.shape[0] == 0:
            traj = np.zeros((1, 13), dtype=float)
            traj[0, 0] = 0.0
            traj[0, 1:4] = p0
            traj[0, 4:7] = v0_parallel
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
            end = State3D(p=p0.copy(), v=v0_parallel.copy(), a=np.zeros(3))
            return SegmentResult(traj=traj, end_state=end, info=info)

        # force end to exactly p1 by shifting s
        s_end = float(samples[-1, 1])
        samples[:, 1] += (L - s_end)

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


class JugglePath:
    """Chains primitives: end state of segment i becomes start state of segment i+1."""

    def __init__(self, start: State3D, sample_hz: float):
        self._start = State3D(start.p, start.v, start.a)
        self._sample_hz = float(sample_hz)
        self._segments: List[Primitive3D] = []

    def add(self, prim: Primitive3D) -> "JugglePath":
        self._segments.append(prim)
        return self

    def build(self) -> PathResult:
        if self._sample_hz <= 0:
            raise ValueError("sample_hz must be > 0")

        all_traj: List[np.ndarray] = []
        infos: List[Dict[str, float]] = []

        cur = State3D(self._start.p.copy(), self._start.v.copy(), self._start.a.copy())
        t_offset = 0.0

        for i, seg in enumerate(self._segments):
            r = seg.generate(start=cur, sample_hz=self._sample_hz)

            traj = r.traj.copy()
            traj[:, 0] += t_offset

            if all_traj and traj.shape[0] > 0 and abs(traj[0, 0] - t_offset) < 1e-12:
                traj = traj[1:, :]

            if traj.shape[0] > 0:
                t_offset = float(traj[-1, 0])

            all_traj.append(traj)
            # Keep info as-is (may contain strings like "mode")
            infos.append({"segment_index": i, **r.info})

            cur = r.end_state  # chain by design

        full = np.vstack(all_traj) if all_traj else np.zeros((0, 13), dtype=float)
        return PathResult(traj=full, segment_infos=infos, end_state=cur)
