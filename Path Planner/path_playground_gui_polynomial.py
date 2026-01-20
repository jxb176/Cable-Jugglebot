import sys
import math
import csv
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QComboBox, QCheckBox,
    QFileDialog, QGroupBox, QTableWidget, QTableWidgetItem, QMessageBox,
    QSpinBox
)

import pyqtgraph as pg

from scipy.optimize import minimize

# ----------------------------
# Models
# ----------------------------
@dataclass
class WorkEnvelope:
    z_min: float
    z_max: float
    cyl_radius: float
    margin: float = 0.0


@dataclass
class Waypoint:
    t: float
    p: np.ndarray  # (3,)
    v: Optional[np.ndarray] = None  # (3,) or None => free/assume 0 for now
    a: Optional[np.ndarray] = None  # (3,) only used for quintic

    # If True, the optimizer is allowed to move this waypoint position (XYZ).
    # Intended usage: subdivision-inserted interior points.
    pos_free: bool = False

    # Seed position used for regularization when pos_free points are optimized.
    # (Only meaningful if pos_free=True.)
    p_seed: Optional[np.ndarray] = None

    def v_filled(self) -> np.ndarray:
        return np.zeros(3) if self.v is None else self.v

    def a_filled(self) -> np.ndarray:
        return np.zeros(3) if self.a is None else self.a


# ----------------------------
# Trajectory primitives
# ----------------------------
def collocation_alphas(n: int, chebyshev: bool = True) -> np.ndarray:
    n = int(max(2, n))
    if chebyshev:
        k = np.arange(n)
        return 0.5 * (1.0 - np.cos(np.pi * k / (n - 1)))
    return np.linspace(0.0, 1.0, n)


def eval_cubic_hermite_1d(p0, v0, p1, v1, T, t):
    if T <= 0:
        raise ValueError("T must be > 0")
    s = float(np.clip(t / T, 0.0, 1.0))

    h00 =  2*s**3 - 3*s**2 + 1
    h10 =    s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 =    s**3 -   s**2
    p = h00*p0 + h10*(T*v0) + h01*p1 + h11*(T*v1)

    dh00 =  6*s**2 - 6*s
    dh10 =  3*s**2 - 4*s + 1
    dh01 = -6*s**2 + 6*s
    dh11 =  3*s**2 - 2*s
    v = (dh00*p0 + dh10*(T*v0) + dh01*p1 + dh11*(T*v1)) / T

    d2h00 = 12*s - 6
    d2h10 =  6*s - 4
    d2h01 = -12*s + 6
    d2h11 =  6*s - 2
    a = (d2h00*p0 + d2h10*(T*v0) + d2h01*p1 + d2h11*(T*v1)) / (T*T)

    j = (12*(p0 - p1) + 6*T*(v0 + v1)) / (T**3)  # constant
    return p, v, a, j


def quintic_coeffs(p0, v0, a0, p1, v1, a1, T) -> np.ndarray:
    T2, T3, T4, T5 = T*T, T*T*T, T*T*T*T, T*T*T*T*T
    c0 = p0
    c1 = v0
    c2 = 0.5 * a0

    A = np.array([
        [T3,   T4,    T5],
        [3*T2, 4*T3,  5*T4],
        [6*T,  12*T2, 20*T3],
    ], dtype=float)

    b = np.array([
        p1 - (c0 + c1*T + c2*T2),
        v1 - (c1 + 2*c2*T),
        a1 - (2*c2),
    ], dtype=float)

    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4, c5], dtype=float)


def eval_quintic_1d(c: np.ndarray, t: float):
    c0, c1, c2, c3, c4, c5 = c
    p = c0 + c1*t + c2*t*t + c3*t**3 + c4*t**4 + c5*t**5
    v = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    a = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
    j = 6*c3 + 24*c4*t + 60*c5*t**2
    return p, v, a, j


# ----------------------------
# Segment evaluation
# ----------------------------
@dataclass
class Segment:
    t0: float
    t1: float
    kind: str  # "cubic" or "quintic"
    # endpoints
    p0: np.ndarray
    p1: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    a0: np.ndarray
    a1: np.ndarray
    # quintic coeffs if used
    c_xyz: Optional[np.ndarray] = None  # (3,6)

    def T(self) -> float:
        return float(self.t1 - self.t0)

    def build_quintic(self):
        T = self.T()
        c = np.zeros((3, 6), dtype=float)
        for k in range(3):
            c[k] = quintic_coeffs(self.p0[k], self.v0[k], self.a0[k], self.p1[k], self.v1[k], self.a1[k], T)
        self.c_xyz = c

    def eval(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (p,v,a,j) 3-vectors at time t (global)."""
        tl = float(np.clip(t - self.t0, 0.0, self.T()))
        T = self.T()
        p = np.zeros(3); v = np.zeros(3); a = np.zeros(3); j = np.zeros(3)

        if self.kind == "cubic":
            for k in range(3):
                pi, vi, ai, ji = eval_cubic_hermite_1d(self.p0[k], self.v0[k], self.p1[k], self.v1[k], T, tl)
                p[k], v[k], a[k], j[k] = pi, vi, ai, ji
            return p, v, a, j

        # quintic
        if self.c_xyz is None:
            self.build_quintic()
        for k in range(3):
            pi, vi, ai, ji = eval_quintic_1d(self.c_xyz[k], tl)
            p[k], v[k], a[k], j[k] = pi, vi, ai, ji
        return p, v, a, j


def build_segments(waypoints: List[Waypoint], kind: str) -> List[Segment]:
    wps = sorted(waypoints, key=lambda w: w.t)
    segs: List[Segment] = []
    for i in range(len(wps) - 1):
        w0, w1 = wps[i], wps[i+1]
        if w1.t <= w0.t:
            raise ValueError("Waypoint times must be strictly increasing.")
        seg = Segment(
            t0=w0.t, t1=w1.t, kind=kind,
            p0=w0.p.copy(), p1=w1.p.copy(),
            v0=w0.v_filled(), v1=w1.v_filled(),
            a0=w0.a_filled(), a1=w1.a_filled(),
        )
        if kind == "quintic":
            seg.build_quintic()
        segs.append(seg)
    return segs


def subdivide_waypoints(wps: List[Waypoint], kind: str, n_sub: int) -> List[Waypoint]:
    """Insert (n_sub-1) intermediate waypoints per segment.

    Positions are sampled from the current piecewise trajectory, so the geometry stays
    consistent. Inserted points have v/a set to None (free knobs for the optimizer).

    Note: this increases the decision variables significantly.
    """
    n_sub = int(max(1, n_sub))
    wps = sorted(wps, key=lambda w: w.t)
    if n_sub == 1:
        return wps

    segs = build_segments(wps, kind=kind)

    out: List[Waypoint] = []
    # keep first
    w0 = wps[0]
    out.append(Waypoint(t=float(w0.t), p=w0.p.copy(),
                        v=None if w0.v is None else w0.v.copy(),
                        a=None if w0.a is None else w0.a.copy(),
                        pos_free=False, p_seed=w0.p.copy()))

    for i, seg in enumerate(segs):
        T = seg.T()
        # insert interior points
        for k in range(1, n_sub):
            a = k / n_sub
            t = float(seg.t0 + a * T)
            p, _v, _a, _j = seg.eval(t)
            # Inserted interior point: position can be optimized.
            out.append(Waypoint(t=t, p=p.copy(), v=None, a=None,
                                pos_free=True, p_seed=p.copy()))

        # keep original endpoint
        we = wps[i+1]
        out.append(Waypoint(t=float(we.t), p=we.p.copy(),
                            v=None if we.v is None else we.v.copy(),
                            a=None if we.a is None else we.a.copy(),
                            pos_free=False, p_seed=we.p.copy()))

    # dedup by time (segment boundaries)
    out2: List[Waypoint] = []
    last_t = None
    for w in sorted(out, key=lambda w: w.t):
        if last_t is not None and abs(w.t - last_t) < 1e-12:
            continue
        out2.append(w)
        last_t = w.t

    # sanity
    for i in range(len(out2)-1):
        if out2[i+1].t <= out2[i].t:
            raise ValueError('Subdivision produced non-increasing waypoint times.')
    return out2

# ----------------
# Optimizer
# ----------------
def sample_trajectory(segs: List[Segment], sample_hz: float) -> np.ndarray:
    dt = 1.0 / float(sample_hz)
    t_start = segs[0].t0
    t_end = segs[-1].t1
    n = int(math.floor((t_end - t_start) / dt))
    ts = [t_start + k*dt for k in range(n)]
    if len(ts) == 0 or abs(ts[-1] - t_end) > 1e-12:
        ts.append(t_end)

    rows = []
    si = 0
    for t in ts:
        while si < len(segs)-1 and t > segs[si].t1:
            si += 1
        p, v, a, j = segs[si].eval(t)
        rows.append([t] + p.tolist() + v.tolist() + a.tolist() + j.tolist())
    return np.array(rows, dtype=float)


def pack_free_velocities(waypoints: List[Waypoint]) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Decision variables = velocities at internal waypoints (excluding first and last),
    3 components each. Returns (x0, index_map) where index_map entries are (wp_i, axis_k).
    """
    idx = []
    x0 = []
    for i in range(1, len(waypoints)-1):
        v = waypoints[i].v_filled()
        for k in range(3):
            idx.append((i, k))
            x0.append(float(v[k]))
    return np.array(x0, dtype=float), idx


def pack_free_positions(waypoints: List[Waypoint]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Decision variables = positions for waypoints with pos_free=True (excluding endpoints).

    Returns (x0, index_map) where index_map entries are (wp_i, axis_k).
    """
    idx: List[Tuple[int, int]] = []
    x0: List[float] = []
    for i in range(1, len(waypoints) - 1):
        w = waypoints[i]
        if not getattr(w, "pos_free", False):
            continue
        for k in range(3):
            idx.append((i, k))
            x0.append(float(w.p[k]))
    return np.array(x0, dtype=float), idx


def apply_free_velocities(waypoints: List[Waypoint], x: np.ndarray, index_map: List[Tuple[int,int]]) -> None:
    """Write decision variables back into waypoints[i].v."""
    # Ensure v arrays exist
    for i in range(len(waypoints)):
        if waypoints[i].v is None:
            waypoints[i].v = np.zeros(3, dtype=float)

    for val, (i, k) in zip(x, index_map):
        waypoints[i].v[k] = float(val)


def apply_free_positions(waypoints: List[Waypoint], x: np.ndarray, index_map: List[Tuple[int, int]]) -> None:
    """Write decision variables back into waypoints[i].p."""
    for val, (i, k) in zip(x, index_map):
        waypoints[i].p[k] = float(val)


def pack_time_deltas(waypoints: List[Waypoint], min_dt: float) -> np.ndarray:
    wps = sorted(waypoints, key=lambda w: w.t)
    dts = np.diff([w.t for w in wps])
    dts = np.maximum(dts, float(min_dt))
    return dts.astype(float)

def apply_time_deltas(waypoints: List[Waypoint], dts: np.ndarray) -> None:
    wps = sorted(waypoints, key=lambda w: w.t)
    t0 = float(wps[0].t)
    t = t0
    wps[0].t = t0
    for i in range(len(dts)):
        t += float(dts[i])
        wps[i+1].t = t


def optimize_velocities_and_times(
    waypoints: List[Waypoint],
    kind: str,
    sample_hz: float,
    env: Optional[WorkEnvelope],
    n_col: int,
    jerk_enable: bool,
    jmax: np.ndarray,
    acc_enable: bool,
    amax: np.ndarray,
    endacc_constrain: bool,
    endamax: np.ndarray,
    endacc_obj_enable: bool,
    endacc_w: float,
    pos_enable: bool,
    pos_w: float,
    v_reg_lambda: float,
    time_w: float,
    jerk_w: float,
    min_seg_dt: float,
    max_iters: int,
) -> Tuple[List[Waypoint], Dict]:

    wps = sorted(waypoints, key=lambda w: w.t)

    x_p0, p_map = pack_free_positions(wps) if pos_enable else (np.zeros(0, dtype=float), [])
    x_v0, v_map = pack_free_velocities(wps)
    x_dt0 = pack_time_deltas(wps, min_dt=min_seg_dt)

    x0 = np.concatenate([x_p0, x_v0, x_dt0], axis=0)

    npv = x_p0.size
    nv = x_v0.size
    ndt = x_dt0.size

    def build_from_x(x: np.ndarray):
        xp = x[:npv]
        xv = x[npv:npv+nv]
        xdt = x[npv+nv:npv+nv+ndt]

        # enforce positivity softly (SLSQP can violate slightly)
        xdt = np.maximum(xdt, float(min_seg_dt))

        wps2 = [Waypoint(t=w.t, p=w.p.copy(),
                         v=None if w.v is None else w.v.copy(),
                         a=None if w.a is None else w.a.copy(),
                         pos_free=getattr(w, "pos_free", False),
                         p_seed=None if w.p_seed is None else w.p_seed.copy())
                for w in wps]
        if pos_enable and npv > 0:
            apply_free_positions(wps2, xp, p_map)
        apply_free_velocities(wps2, xv, v_map)
        apply_time_deltas(wps2, xdt)

        segs2 = build_segments(wps2, kind=kind)
        return wps2, segs2, xdt

    def objective(x: np.ndarray) -> float:
        wps2, segs2, xdt = build_from_x(x)
        traj = sample_trajectory(segs2, sample_hz=sample_hz)
        J = jerk_cost(traj)

        Ttotal = float(np.sum(xdt))

        # Regularize free positions toward their seeds (keeps the optimizer from wandering)
        Pdev = 0.0
        if pos_enable and pos_w > 0.0 and npv > 0:
            for (i, _k) in set([(ii, kk) for (ii, kk) in p_map]):
                w = wps2[i]
                if w.p_seed is None:
                    continue
                d = (w.p - w.p_seed)
                Pdev += float(np.dot(d, d))

        Vreg = float(v_reg_lambda) * float(np.dot(x[npv:npv+nv], x[npv:npv+nv]))

        Eacc = 0.0
        if endacc_obj_enable and endacc_w > 0.0:
            Eacc = float(endacc_w) * endpoint_accel_cost(segs2)

        return float(time_w * Ttotal + jerk_w * J + Vreg + float(pos_w) * Pdev + Eacc)

    cons = []

    if env is not None:
        def env_margin_min(x: np.ndarray) -> float:
            _wps2, segs2, _xdt = build_from_x(x)
            ok, worst = check_envelope_collocation(segs2, env, n_col=n_col, chebyshev=True)
            return float(min(worst["zmin"][0], worst["zmax"][0], worst["rad"][0]))
        cons.append({"type": "ineq", "fun": env_margin_min})

    if jerk_enable:
        def jerk_margin_min(x: np.ndarray) -> float:
            _wps2, segs2, _xdt = build_from_x(x)
            ok, worst = check_jerk_limit(segs2, jmax=jmax, n_col=n_col)
            return float(min(worst["jx"][0], worst["jy"][0], worst["jz"][0]))
        cons.append({"type": "ineq", "fun": jerk_margin_min})


    if acc_enable:
        def accel_margin_min(x: np.ndarray) -> float:
            _wps2, segs2, _xdt = build_from_x(x)
            ok, worst = check_accel_limit(segs2, amax=amax, n_col=n_col)
            return float(min(worst["ax"][0], worst["ay"][0], worst["az"][0]))
        cons.append({"type": "ineq", "fun": accel_margin_min})

    if endacc_constrain:
        def endacc_margin_min(x: np.ndarray) -> float:
            _wps2, segs2, _xdt = build_from_x(x)
            ok, worst = check_endpoint_accel_limit(segs2, amax_end=endamax)
            return float(min(worst["ax0"][0], worst["ay0"][0], worst["az0"][0],
                             worst["ax1"][0], worst["ay1"][0], worst["az1"][0]))
        cons.append({"type": "ineq", "fun": endacc_margin_min})

    # Also enforce dt >= min_seg_dt explicitly as constraints
    for i in range(ndt):
        def dt_i_fun(x, i=i):
            return float(x[npv + nv + i] - min_seg_dt)
        cons.append({"type": "ineq", "fun": dt_i_fun})

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=cons,
        options={"maxiter": int(max_iters), "ftol": 1e-9, "disp": False},
    )

    wps_opt, segs_opt, dts_opt = build_from_x(res.x)
    traj_opt = sample_trajectory(segs_opt, sample_hz=sample_hz)

    info = {
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", 0)),
        "cost": float(jerk_cost(traj_opt)),
        "T_total": float(np.sum(dts_opt)),
        "objective": float(objective(res.x)),
    }

    if env is not None:
        ok_env, worst_env = check_envelope_collocation(segs_opt, env, n_col=n_col, chebyshev=True)
        info["env_ok"] = bool(ok_env)
        info["env_worst"] = worst_env
    if jerk_enable:
        ok_j, worst_j = check_jerk_limit(segs_opt, jmax=jmax, n_col=n_col)
        info["jerk_ok"] = bool(ok_j)
        info["jerk_worst"] = worst_j

    if acc_enable:
        ok_a, worst_a = check_accel_limit(segs_opt, amax=amax, n_col=n_col)
        info["acc_ok"] = bool(ok_a)
        info["acc_worst"] = worst_a

    if endacc_constrain or endacc_obj_enable:
        ok_e, worst_e = check_endpoint_accel_limit(segs_opt, amax_end=endamax)
        info["endacc_ok"] = bool(ok_e)
        info["endacc_worst"] = worst_e
        info["endacc_cost"] = float(endpoint_accel_cost(segs_opt))

    return wps_opt, info



def optimize_waypoint_velocities(
    waypoints: List[Waypoint],
    kind: str,
    sample_hz: float,
    env: Optional[WorkEnvelope],
    n_col: int,
    jerk_enable: bool,
    jmax: np.ndarray,
    acc_enable: bool,
    amax: np.ndarray,
    endacc_constrain: bool,
    endamax: np.ndarray,
    endacc_obj_enable: bool,
    endacc_w: float,
    pos_enable: bool,
    pos_w: float,
    v_reg_lambda: float,
    max_iters: int,
) -> Tuple[List[Waypoint], Dict]:
    """
    Optimize internal waypoint velocities (XYZ) for piecewise segments to minimize jerk^2
    while satisfying envelope collocation constraints and optional jerk limit.

    V1 design:
      - positions + times fixed
      - only velocities at internal waypoints are decision variables
      - first/last waypoint velocities kept as-is
      - cubic supported (recommended); quintic also supported but this optimizer only edits v
    """

    wps = sorted(waypoints, key=lambda w: w.t)
    x_p0, p_map = pack_free_positions(wps) if pos_enable else (np.zeros(0, dtype=float), [])
    x_v0, v_map = pack_free_velocities(wps)
    x0 = np.concatenate([x_p0, x_v0], axis=0)
    npv = x_p0.size
    nv = x_v0.size

    # Precompute: if no internal waypoints, nothing to optimize
    if nv == 0 and npv == 0:
        segs = build_segments(wps, kind=kind)
        traj = sample_trajectory(segs, sample_hz=sample_hz)
        return wps, {"success": True, "message": "No internal waypoint velocities to optimize.", "cost": jerk_cost(traj)}

    def build_from_x(x: np.ndarray):
        xp = x[:npv]
        xv = x[npv:npv+nv]

        wps2 = [Waypoint(t=w.t, p=w.p.copy(),
                         v=None if w.v is None else w.v.copy(),
                         a=None if w.a is None else w.a.copy(),
                         pos_free=getattr(w, "pos_free", False),
                         p_seed=None if w.p_seed is None else w.p_seed.copy())
                for w in wps]
        if pos_enable and npv > 0:
            apply_free_positions(wps2, xp, p_map)
        apply_free_velocities(wps2, xv, v_map)
        segs2 = build_segments(wps2, kind=kind)
        return wps2, segs2

    def objective(x: np.ndarray) -> float:
        wps2, segs2 = build_from_x(x)
        traj = sample_trajectory(segs2, sample_hz=sample_hz)
        cost = jerk_cost(traj)

        # Regularize free positions toward their seeds
        if pos_enable and pos_w > 0.0 and npv > 0:
            Pdev = 0.0
            for (i, _k) in set([(ii, kk) for (ii, kk) in p_map]):
                w = wps2[i]
                if w.p_seed is None:
                    continue
                d = (w.p - w.p_seed)
                Pdev += float(np.dot(d, d))
            cost += float(pos_w) * Pdev

        if v_reg_lambda > 0.0 and nv > 0:
            cost += float(v_reg_lambda) * float(np.dot(x[npv:npv+nv], x[npv:npv+nv]))

        if endacc_obj_enable and endacc_w > 0.0:
            cost += float(endacc_w) * endpoint_accel_cost(segs2)

        return float(cost)

    # ---- Constraints for SLSQP must be >= 0 ----
    cons = []

    if env is not None:
        def env_margin_min(x: np.ndarray) -> float:
            _wps2, segs2 = build_from_x(x)
            ok, worst = check_envelope_collocation(segs2, env, n_col=n_col, chebyshev=True)
            # Return the minimum margin across zmin/zmax/rad
            return float(min(worst["zmin"][0], worst["zmax"][0], worst["rad"][0]))
        cons.append({"type": "ineq", "fun": env_margin_min})

    if jerk_enable:
        def jerk_margin_min(x: np.ndarray) -> float:
            _wps2, segs2 = build_from_x(x)
            ok, worst = check_jerk_limit(segs2, jmax=jmax, n_col=n_col)
            return float(min(worst["jx"][0], worst["jy"][0], worst["jz"][0]))
        cons.append({"type": "ineq", "fun": jerk_margin_min})


    if acc_enable:
        def accel_margin_min(x: np.ndarray) -> float:
            _wps2, segs2 = build_from_x(x)
            ok, worst = check_accel_limit(segs2, amax=amax, n_col=n_col)
            return float(min(worst["ax"][0], worst["ay"][0], worst["az"][0]))
        cons.append({"type": "ineq", "fun": accel_margin_min})

    if endacc_constrain:
        def endacc_margin_min(x: np.ndarray) -> float:
            _wps2, segs2 = build_from_x(x)
            ok, worst = check_endpoint_accel_limit(segs2, amax_end=endamax)
            return float(min(worst["ax0"][0], worst["ay0"][0], worst["az0"][0],
                             worst["ax1"][0], worst["ay1"][0], worst["az1"][0]))
        cons.append({"type": "ineq", "fun": endacc_margin_min})

    # Solve
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=cons,
        options={
            "maxiter": int(max_iters),
            "ftol": 1e-9,
            "disp": False,
        },
    )

    wps_opt, segs_opt = build_from_x(res.x)
    traj_opt = sample_trajectory(segs_opt, sample_hz=sample_hz)

    info = {
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", 0)),
        "cost": float(jerk_cost(traj_opt)),
        "cost_with_reg": float(objective(res.x)),
    }

    # Add constraint diagnostics
    if env is not None:
        ok_env, worst_env = check_envelope_collocation(segs_opt, env, n_col=n_col, chebyshev=True)
        info["env_ok"] = bool(ok_env)
        info["env_worst"] = worst_env
    if jerk_enable:
        ok_j, worst_j = check_jerk_limit(segs_opt, jmax=jmax, n_col=n_col)
        info["jerk_ok"] = bool(ok_j)
        info["jerk_worst"] = worst_j

    if acc_enable:
        ok_a, worst_a = check_accel_limit(segs_opt, amax=amax, n_col=n_col)
        info["acc_ok"] = bool(ok_a)
        info["acc_worst"] = worst_a

    if endacc_constrain or endacc_obj_enable:
        ok_e, worst_e = check_endpoint_accel_limit(segs_opt, amax_end=endamax)
        info["endacc_ok"] = bool(ok_e)
        info["endacc_worst"] = worst_e
        info["endacc_cost"] = float(endpoint_accel_cost(segs_opt))

    return wps_opt, info


# ----------------------------
# Constraints
# ----------------------------
def check_envelope_collocation(segs: List[Segment], env: WorkEnvelope, n_col: int, chebyshev: bool = True):
    R = max(0.0, env.cyl_radius - env.margin)
    zmin = env.z_min + env.margin
    zmax = env.z_max - env.margin

    worst = {
        "zmin": (+np.inf, None, None),
        "zmax": (+np.inf, None, None),
        "rad":  (+np.inf, None, None),
    }

    for seg in segs:
        T = seg.T()
        for a in collocation_alphas(n_col, chebyshev=chebyshev):
            t = seg.t0 + float(a)*T
            p, _, _, _ = seg.eval(t)
            x, y, z = float(p[0]), float(p[1]), float(p[2])

            m_zmin = z - zmin
            m_zmax = zmax - z
            m_rad  = (R*R) - (x*x + y*y)

            if m_zmin < worst["zmin"][0]:
                worst["zmin"] = (float(m_zmin), t, (x,y,z))
            if m_zmax < worst["zmax"][0]:
                worst["zmax"] = (float(m_zmax), t, (x,y,z))
            if m_rad < worst["rad"][0]:
                worst["rad"] = (float(m_rad), t, (x,y,z))

    ok = (worst["zmin"][0] >= 0.0) and (worst["zmax"][0] >= 0.0) and (worst["rad"][0] >= 0.0)
    return ok, worst


def check_jerk_limit(segs: List[Segment], jmax: np.ndarray, n_col: int):
    """Check |j| <= jmax component-wise at collocation points (works for quintic too)."""
    worst = {
        "jx": (+np.inf, None, None),
        "jy": (+np.inf, None, None),
        "jz": (+np.inf, None, None),
    }
    for seg in segs:
        T = seg.T()
        for a in collocation_alphas(n_col, chebyshev=True):
            t = seg.t0 + float(a)*T
            _, _, _, j = seg.eval(t)
            for k, key in enumerate(["jx", "jy", "jz"]):
                margin = float(jmax[k] - abs(j[k]))
                if margin < worst[key][0]:
                    worst[key] = (margin, t, float(j[k]))
    ok = all(worst[k][0] >= 0.0 for k in worst.keys())
    return ok, worst



def check_accel_limit(segs: List[Segment], amax: np.ndarray, n_col: int):
    """Check |a| <= amax component-wise at collocation points."""
    worst = {
        "ax": (+np.inf, None, None),
        "ay": (+np.inf, None, None),
        "az": (+np.inf, None, None),
    }
    for seg in segs:
        T = seg.T()
        for a in collocation_alphas(n_col, chebyshev=True):
            t = seg.t0 + float(a) * T
            _p, _v, acc, _j = seg.eval(t)
            for k, key in enumerate(["ax", "ay", "az"]):
                margin = float(amax[k] - abs(acc[k]))
                if margin < worst[key][0]:
                    worst[key] = (margin, t, float(acc[k]))
    ok = all(worst[k][0] >= 0.0 for k in worst.keys())
    return ok, worst


def endpoint_accels(segs: List[Segment]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (a_start, a_end) accelerations at the trajectory endpoints."""
    if not segs:
        return np.zeros(3), np.zeros(3)
    _p0, _v0, a0, _j0 = segs[0].eval(segs[0].t0)
    _p1, _v1, a1, _j1 = segs[-1].eval(segs[-1].t1)
    return a0, a1


def check_endpoint_accel_limit(segs: List[Segment], amax_end: np.ndarray):
    """Check |a_start| and |a_end| <= amax_end component-wise. Returns (ok, worst)."""
    a0, a1 = endpoint_accels(segs)
    worst = {
        "ax0": (+float('inf'), None, None),
        "ay0": (+float('inf'), None, None),
        "az0": (+float('inf'), None, None),
        "ax1": (+float('inf'), None, None),
        "ay1": (+float('inf'), None, None),
        "az1": (+float('inf'), None, None),
    }

    # start
    for k, key in enumerate(["ax0", "ay0", "az0"]):
        margin = float(amax_end[k] - abs(a0[k]))
        worst[key] = (margin, float(segs[0].t0), float(a0[k]))

    # end
    for k, key in enumerate(["ax1", "ay1", "az1"]):
        margin = float(amax_end[k] - abs(a1[k]))
        worst[key] = (margin, float(segs[-1].t1), float(a1[k]))

    ok = all(v[0] >= 0.0 for v in worst.values())
    return ok, worst


def endpoint_accel_cost(segs: List[Segment]) -> float:
    """Return ||a_start||^2 + ||a_end||^2."""
    a0, a1 = endpoint_accels(segs)
    return float(np.dot(a0, a0) + np.dot(a1, a1))


def jerk_cost(traj: np.ndarray) -> float:
    # approximate integral of ||j||^2 dt
    t = traj[:, 0]
    j = traj[:, 10:13]
    dt = np.diff(t)
    if len(dt) == 0:
        return 0.0
    # trapezoid
    jj = np.sum(j*j, axis=1)
    return float(np.sum(0.5*(jj[:-1] + jj[1:]) * dt))


def format_worst_margin(name, entry):
    m, t, xyz = entry
    if t is None:
        return f"{name}: n/a"
    if isinstance(xyz, tuple) and len(xyz) == 3:
        x, y, z = xyz
        return f"{name}: margin={m:+.6f} at t={t:.4f}s xyz=({x:+.4f},{y:+.4f},{z:+.4f})"
    return f"{name}: margin={m:+.6f} at t={t:.4f}s value={xyz}"


# ----------------------------
# GUI
# ----------------------------
class PathPlayground(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Task-Space Path Playground (XYZ) - Cubic/Quintic + Envelope + Jerk")

        self.waypoints: List[Waypoint] = self.default_waypoints()

        # Main layout
        root = QHBoxLayout(self)

        # Left: controls + waypoint table
        left = QVBoxLayout()
        root.addLayout(left, 0)

        # Right: plots
        right = QVBoxLayout()
        root.addLayout(right, 1)

        # Controls group
        ctrl = QGroupBox("Controls")
        left.addWidget(ctrl)
        g = QGridLayout(ctrl)

        row = 0
        g.addWidget(QLabel("Segment type:"), row, 0)
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(["cubic", "quintic"])
        g.addWidget(self.kind_combo, row, 1)
        row += 1

        g.addWidget(QLabel("Sample Hz:"), row, 0)
        self.sample_hz = QDoubleSpinBox()
        self.sample_hz.setRange(10.0, 5000.0)
        self.sample_hz.setDecimals(1)
        self.sample_hz.setValue(200.0)
        g.addWidget(self.sample_hz, row, 1)
        row += 1

        g.addWidget(QLabel("Collocation pts/seg:"), row, 0)
        self.ncol = QSpinBox()
        self.ncol.setRange(2, 200)
        self.ncol.setValue(10)
        g.addWidget(self.ncol, row, 1)
        row += 1

        g.addWidget(QLabel("Subdivide each segment:"), row, 0)
        self.subdivide = QSpinBox()
        self.subdivide.setRange(1, 50)
        self.subdivide.setValue(1)
        g.addWidget(self.subdivide, row, 1)
        row += 1

        # Envelope group
        envg = QGroupBox("Envelope (cylinder)")
        left.addWidget(envg)
        eg = QGridLayout(envg)

        self.env_enable = QCheckBox("Enable envelope")
        self.env_enable.setChecked(True)
        eg.addWidget(self.env_enable, 0, 0, 1, 2)

        eg.addWidget(QLabel("z_min [m]"), 1, 0)
        self.zmin = QDoubleSpinBox(); self.zmin.setRange(-10, 10); self.zmin.setDecimals(4); self.zmin.setValue(-0.30)
        eg.addWidget(self.zmin, 1, 1)

        eg.addWidget(QLabel("z_max [m]"), 2, 0)
        self.zmax = QDoubleSpinBox(); self.zmax.setRange(-10, 10); self.zmax.setDecimals(4); self.zmax.setValue(0.30)
        eg.addWidget(self.zmax, 2, 1)

        eg.addWidget(QLabel("radius [m]"), 3, 0)
        self.rad = QDoubleSpinBox(); self.rad.setRange(0, 10); self.rad.setDecimals(4); self.rad.setValue(0.25)
        eg.addWidget(self.rad, 3, 1)

        eg.addWidget(QLabel("margin [m]"), 4, 0)
        self.margin = QDoubleSpinBox(); self.margin.setRange(0, 1); self.margin.setDecimals(4); self.margin.setValue(0.00)
        eg.addWidget(self.margin, 4, 1)

        # Jerk group
        jg = QGroupBox("Jerk limits (component-wise)")
        left.addWidget(jg)
        jgl = QGridLayout(jg)
        self.jerk_enable = QCheckBox("Enable jerk limit")
        self.jerk_enable.setChecked(False)
        jgl.addWidget(self.jerk_enable, 0, 0, 1, 2)

        jgl.addWidget(QLabel("jmax_x [m/s^3]"), 1, 0)
        self.jmax_x = QDoubleSpinBox(); self.jmax_x.setRange(0, 1e6); self.jmax_x.setDecimals(3); self.jmax_x.setValue(200.0)
        jgl.addWidget(self.jmax_x, 1, 1)

        jgl.addWidget(QLabel("jmax_y [m/s^3]"), 2, 0)
        self.jmax_y = QDoubleSpinBox(); self.jmax_y.setRange(0, 1e6); self.jmax_y.setDecimals(3); self.jmax_y.setValue(200.0)
        jgl.addWidget(self.jmax_y, 2, 1)

        jgl.addWidget(QLabel("jmax_z [m/s^3]"), 3, 0)
        self.jmax_z = QDoubleSpinBox(); self.jmax_z.setRange(0, 1e6); self.jmax_z.setDecimals(3); self.jmax_z.setValue(200.0)
        jgl.addWidget(self.jmax_z, 3, 1)


        # Accel group
        ag = QGroupBox("Acceleration limits (component-wise)")
        left.addWidget(ag)
        agl = QGridLayout(ag)

        self.acc_enable = QCheckBox("Enable accel limit")
        self.acc_enable.setChecked(False)
        agl.addWidget(self.acc_enable, 0, 0, 1, 2)

        agl.addWidget(QLabel("amax_x [m/s^2]"), 1, 0)
        self.amax_x = QDoubleSpinBox(); self.amax_x.setRange(0, 1e6); self.amax_x.setDecimals(3); self.amax_x.setValue(50.0)
        agl.addWidget(self.amax_x, 1, 1)

        agl.addWidget(QLabel("amax_y [m/s^2]"), 2, 0)
        self.amax_y = QDoubleSpinBox(); self.amax_y.setRange(0, 1e6); self.amax_y.setDecimals(3); self.amax_y.setValue(50.0)
        agl.addWidget(self.amax_y, 2, 1)

        agl.addWidget(QLabel("amax_z [m/s^2]"), 3, 0)
        self.amax_z = QDoubleSpinBox(); self.amax_z.setRange(0, 1e6); self.amax_z.setDecimals(3); self.amax_z.setValue(50.0)
        agl.addWidget(self.amax_z, 3, 1)

        # Endpoint accel constraint (start/end)
        self.endacc_enable = QCheckBox("Constrain endpoint accel")
        self.endacc_enable.setChecked(False)
        agl.addWidget(self.endacc_enable, 4, 0, 1, 2)

        agl.addWidget(QLabel("end amax_x [m/s^2]"), 5, 0)
        self.endamax_x = QDoubleSpinBox(); self.endamax_x.setRange(0, 1e6); self.endamax_x.setDecimals(3); self.endamax_x.setValue(5.0)
        agl.addWidget(self.endamax_x, 5, 1)

        agl.addWidget(QLabel("end amax_y [m/s^2]"), 6, 0)
        self.endamax_y = QDoubleSpinBox(); self.endamax_y.setRange(0, 1e6); self.endamax_y.setDecimals(3); self.endamax_y.setValue(5.0)
        agl.addWidget(self.endamax_y, 6, 1)

        agl.addWidget(QLabel("end amax_z [m/s^2]"), 7, 0)
        self.endamax_z = QDoubleSpinBox(); self.endamax_z.setRange(0, 1e6); self.endamax_z.setDecimals(3); self.endamax_z.setValue(5.0)
        agl.addWidget(self.endamax_z, 7, 1)

        # Optimization group
        cg = QGroupBox("Optimizer")
        left.addWidget(cg)
        cgl = QGridLayout(cg)

        # Objectives (checked => included in objective)
        self.obj_time = QCheckBox("Minimize time")
        self.obj_time.setChecked(True)
        cgl.addWidget(self.obj_time, 0, 0, 1, 2)

        self.obj_jerk = QCheckBox("Minimize jerk^2")
        self.obj_jerk.setChecked(True)
        cgl.addWidget(self.obj_jerk, 1, 0, 1, 2)

        self.obj_endacc = QCheckBox("Minimize endpoint accel^2")
        self.obj_endacc.setChecked(False)
        cgl.addWidget(self.obj_endacc, 2, 0, 1, 2)

        cgl.addWidget(QLabel("End-acc weight δ"), 3, 0)
        self.endacc_w = QDoubleSpinBox()
        self.endacc_w.setRange(0.0, 1e9)
        self.endacc_w.setDecimals(6)
        self.endacc_w.setValue(1.0)
        cgl.addWidget(self.endacc_w, 3, 1)

        cgl.addWidget(QLabel("Opt max iters"), 4, 0)
        self.opt_max_iters = QSpinBox()
        self.opt_max_iters.setRange(1, 1000)
        self.opt_max_iters.setValue(200)
        cgl.addWidget(self.opt_max_iters, 4, 1)

        cgl.addWidget(QLabel("Velocity reg (λ)"), 5, 0)
        self.opt_v_reg = QDoubleSpinBox()
        self.opt_v_reg.setRange(0.0, 1e6)
        self.opt_v_reg.setDecimals(6)
        self.opt_v_reg.setValue(0.0)
        cgl.addWidget(self.opt_v_reg, 5, 1)

        # Optional: allow optimizer to move subdivision-inserted interior waypoint positions.
        self.pos_enable = QCheckBox("Optimize interior positions")
        self.pos_enable.setChecked(False)
        cgl.addWidget(self.pos_enable, 6, 0, 1, 2)

        cgl.addWidget(QLabel("Pos deviation weight γ"), 7, 0)
        self.pos_w = QDoubleSpinBox()
        self.pos_w.setRange(0.0, 1e9)
        self.pos_w.setDecimals(6)
        self.pos_w.setValue(1.0)
        cgl.addWidget(self.pos_w, 7, 1)

        cgl.addWidget(QLabel("Time weight α"), 8, 0)
        self.time_w = QDoubleSpinBox()
        self.time_w.setRange(0.0, 1e6)
        self.time_w.setDecimals(6)
        self.time_w.setValue(1.0)
        cgl.addWidget(self.time_w, 8, 1)

        cgl.addWidget(QLabel("Jerk weight β"), 9, 0)
        self.jerk_w = QDoubleSpinBox()
        self.jerk_w.setRange(0.0, 1e6)
        self.jerk_w.setDecimals(6)
        self.jerk_w.setValue(1.0)
        cgl.addWidget(self.jerk_w, 9, 1)

        cgl.addWidget(QLabel("Min segment dt [s]"), 10, 0)
        self.min_seg_dt = QDoubleSpinBox()
        self.min_seg_dt.setRange(1e-4, 10.0)
        self.min_seg_dt.setDecimals(6)
        self.min_seg_dt.setValue(0.02)
        cgl.addWidget(self.min_seg_dt, 10, 1)

        # Buttons
        btns = QHBoxLayout()
        left.addLayout(btns)

        self.btn_rebuild = QPushButton("Rebuild & Plot")
        self.btn_export = QPushButton("Export CSV…")
        self.btn_reset = QPushButton("Reset waypoints")
        btns.addWidget(self.btn_rebuild)
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_reset)

        # Waypoint table
        tblg = QGroupBox("Waypoints (XYZ)  |  Leave v/a blank = 0 for now")
        left.addWidget(tblg, 1)
        tl = QVBoxLayout(tblg)

        self.table = QTableWidget()
        self.table.setColumnCount(1 + 3 + 3 + 3)
        self.table.setHorizontalHeaderLabels([
            "t",
            "px","py","pz",
            "vx","vy","vz",
            "ax","ay","az",
        ])
        tl.addWidget(self.table)

        self.btn_add = QPushButton("Add waypoint")
        self.btn_del = QPushButton("Delete selected")
        bb = QHBoxLayout()
        bb.addWidget(self.btn_add)
        bb.addWidget(self.btn_del)
        tl.addLayout(bb)

        # Status box
        self.status = QLabel("")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        # Plots (right)
        pg.setConfigOptions(antialias=True)
        self.plot_p = pg.PlotWidget(title="Position XYZ")
        self.plot_v = pg.PlotWidget(title="Velocity XYZ")
        self.plot_a = pg.PlotWidget(title="Acceleration XYZ")
        self.plot_j = pg.PlotWidget(title="Jerk magnitude ||j|| and components")

        right.addWidget(self.plot_p, 1)
        right.addWidget(self.plot_v, 1)
        right.addWidget(self.plot_a, 1)
        right.addWidget(self.plot_j, 1)

        # Wiring
        self.btn_rebuild.clicked.connect(self.rebuild_and_plot)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_reset.clicked.connect(self.reset_waypoints)
        self.btn_add.clicked.connect(self.add_waypoint)
        self.btn_del.clicked.connect(self.delete_selected)

        # populate table + initial plot
        self.populate_table()
        QTimer.singleShot(0, self.rebuild_and_plot)

        self.last_traj: Optional[np.ndarray] = None

    def default_waypoints(self) -> List[Waypoint]:
        # Simple throw-like scaffold: hold at origin -> bottom -> release -> settle -> hold
        # (You can edit in the table)
        return [
            Waypoint(t=0.00, p=np.array([0.0, 0.0, 0.0]), v=np.array([0.0, 0.0, -3.0]), a=np.array([0.0, 0.0, 0.0])),
            Waypoint(t=0.5, p=np.array([0.0, 0.0, -0.20]), v=np.array([0.0, 0.0, 5.0]), a=np.array([0.0, 0.0, 0.0])),
        ]

    def populate_table(self):
        self.table.setRowCount(len(self.waypoints))
        for i, w in enumerate(self.waypoints):
            self._set_cell(i, 0, w.t)
            for k in range(3):
                self._set_cell(i, 1+k, float(w.p[k]))
            v = w.v if w.v is not None else np.zeros(3)
            for k in range(3):
                self._set_cell(i, 4+k, float(v[k]))
            a = w.a if w.a is not None else np.zeros(3)
            for k in range(3):
                self._set_cell(i, 7+k, float(a[k]))

        self.table.resizeColumnsToContents()

    def _set_cell(self, r, c, val: float):
        item = QTableWidgetItem(f"{val:.6f}")
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.setItem(r, c, item)

    def read_table_waypoints(self) -> List[Waypoint]:
        wps: List[Waypoint] = []
        for r in range(self.table.rowCount()):
            def getf(c, default=0.0):
                it = self.table.item(r, c)
                if it is None:
                    return default
                s = it.text().strip()
                if s == "":
                    return default
                return float(s)

            t = getf(0)
            p = np.array([getf(1), getf(2), getf(3)], dtype=float)
            v = np.array([getf(4), getf(5), getf(6)], dtype=float)
            a = np.array([getf(7), getf(8), getf(9)], dtype=float)

            # For cubic, we'll use v; for quintic we use both v and a.
            wps.append(Waypoint(t=t, p=p, v=v, a=a))
        # sort and ensure unique increasing times
        wps = sorted(wps, key=lambda w: w.t)
        # dedup exact times
        out = []
        last_t = None
        for w in wps:
            if last_t is not None and abs(w.t - last_t) < 1e-12:
                continue
            out.append(w)
            last_t = w.t
        if len(out) < 2:
            raise ValueError("Need at least two waypoints with increasing time.")
        for i in range(len(out)-1):
            if out[i+1].t <= out[i].t:
                raise ValueError("Waypoint times must be strictly increasing.")
        return out

    def current_envelope(self) -> WorkEnvelope:
        return WorkEnvelope(
            z_min=float(self.zmin.value()),
            z_max=float(self.zmax.value()),
            cyl_radius=float(self.rad.value()),
            margin=float(self.margin.value())
        )




    def rebuild_and_plot(self):
        """Rebuild segments, optionally run optimizer, then sample + plot."""
        try:
            n_sub = 1
            if hasattr(self, "subdivide"):
                n_sub = int(self.subdivide.value())

            # --- Read inputs ---
            self.waypoints = self.read_table_waypoints()
            kind = self.kind_combo.currentText()

            # Optional segment subdivision (adds intermediate waypoints as extra DOF)
            work_wps = self.waypoints
            if n_sub > 1:
                work_wps = subdivide_waypoints(work_wps, kind=kind, n_sub=n_sub)

            sample_hz = float(self.sample_hz.value())
            ncol = int(self.ncol.value())

            # Independent constraint toggles
            env = self.current_envelope() if self.env_enable.isChecked() else None

            jerk_on = self.jerk_enable.isChecked()
            jmax = np.array([
                float(self.jmax_x.value()),
                float(self.jmax_y.value()),
                float(self.jmax_z.value()),
            ], dtype=float)

            acc_on = self.acc_enable.isChecked()
            amax = np.array([
                float(self.amax_x.value()),
                float(self.amax_y.value()),
                float(self.amax_z.value()),
            ], dtype=float)

            endacc_on = self.endacc_enable.isChecked()
            endamax = np.array([
                float(self.endamax_x.value()),
                float(self.endamax_y.value()),
                float(self.endamax_z.value()),
            ], dtype=float)

            endacc_obj_on = self.obj_endacc.isChecked()
            endacc_w = float(self.endacc_w.value())

            # Objective toggles
            min_time = self.obj_time.isChecked()
            min_jerk = self.obj_jerk.isChecked()

            # Optimizer knobs
            v_reg = float(self.opt_v_reg.value())
            max_iters = int(self.opt_max_iters.value())
            min_seg_dt = float(self.min_seg_dt.value())

            # Optional: optimize subdivision-inserted interior positions
            pos_enable = bool(self.pos_enable.isChecked())
            pos_w = float(self.pos_w.value())

            opt_info = None

            # --- Helper: evaluate constraints for display ---
            def evaluate_constraints(segs_local) -> Tuple[bool, List[str]]:
                msgs: List[str] = []
                ok_all = True

                if env is not None:
                    ok_env, worst_env = check_envelope_collocation(segs_local, env, n_col=ncol, chebyshev=True)
                    if not ok_env:
                        ok_all = False
                        msgs.append(
                            "Envelope violation\n" +
                            "\n".join([
                                format_worst_margin("zmin", worst_env["zmin"]),
                                format_worst_margin("zmax", worst_env["zmax"]),
                                format_worst_margin("rad",  worst_env["rad"]),
                            ])
                        )

                if jerk_on:
                    ok_j, worst_j = check_jerk_limit(segs_local, jmax=jmax, n_col=ncol)
                    if not ok_j:
                        ok_all = False
                        msgs.append(
                            "Jerk violation\n" +
                            "\n".join([
                                format_worst_margin("jx", worst_j["jx"]),
                                format_worst_margin("jy", worst_j["jy"]),
                                format_worst_margin("jz", worst_j["jz"]),
                            ])
                        )


                if acc_on:
                    ok_a, worst_a = check_accel_limit(segs_local, amax=amax, n_col=ncol)
                    if not ok_a:
                        ok_all = False
                        msgs.append(
                            "Accel violation\n" +
                            "\n".join([
                                format_worst_margin("ax", worst_a["ax"]),
                                format_worst_margin("ay", worst_a["ay"]),
                                format_worst_margin("az", worst_a["az"]),
                            ])
                        )

                if endacc_on:
                    ok_e, worst_e = check_endpoint_accel_limit(segs_local, amax_end=endamax)
                    if not ok_e:
                        ok_all = False
                        msgs.append(
                            "Endpoint accel violation\n" +
                            "\n".join([
                                format_worst_margin("ax0", worst_e["ax0"]),
                                format_worst_margin("ay0", worst_e["ay0"]),
                                format_worst_margin("az0", worst_e["az0"]),
                                format_worst_margin("ax1", worst_e["ax1"]),
                                format_worst_margin("ay1", worst_e["ay1"]),
                                format_worst_margin("az1", worst_e["az1"]),
                            ])
                        )

                return ok_all, msgs

            # --- Optional optimization ---
            # Behavior:
            #   - If Minimize time is checked => optimize dt + internal waypoint velocities.
            #   - Else if only Minimize jerk^2 is checked => optimize internal velocities with fixed waypoint times.
            #   - If neither is checked => no optimization; just evaluate the current table.
            if min_time:
                time_w = float(self.time_w.value())
                jerk_w = float(self.jerk_w.value()) if min_jerk else 0.0
                work_wps, opt_info = optimize_velocities_and_times(
                    waypoints=work_wps,
                    kind=kind,
                    sample_hz=sample_hz,
                    env=env,
                    n_col=ncol,
                    jerk_enable=jerk_on,
                    jmax=jmax,
                    acc_enable=acc_on,
                    amax=amax,
                    endacc_constrain=endacc_on,
                    endamax=endamax,
                    endacc_obj_enable=endacc_obj_on,
                    endacc_w=endacc_w,
                    pos_enable=pos_enable,
                    pos_w=pos_w,
                    v_reg_lambda=v_reg,
                    time_w=time_w,
                    jerk_w=jerk_w,
                    min_seg_dt=min_seg_dt,
                    max_iters=max_iters,
                )
                if n_sub == 1:
                    self.waypoints = work_wps
                    self.populate_table()
                else:
                    # keep user table intact when subdividing
                    pass

            elif min_jerk:
                work_wps, opt_info = optimize_waypoint_velocities(
                    waypoints=work_wps,
                    kind=kind,
                    sample_hz=sample_hz,
                    env=env,
                    n_col=ncol,
                    jerk_enable=jerk_on,
                    jmax=jmax,
                    acc_enable=acc_on,
                    amax=amax,
                    endacc_constrain=endacc_on,
                    endamax=endamax,
                    endacc_obj_enable=endacc_obj_on,
                    endacc_w=endacc_w,
                    pos_enable=pos_enable,
                    pos_w=pos_w,
                    v_reg_lambda=v_reg,
                    max_iters=max_iters,
                )
                if n_sub == 1:
                    self.waypoints = work_wps
                    self.populate_table()
                else:
                    pass

            # --- Build final trajectory ---
            segs = build_segments(work_wps, kind=kind)
            traj = sample_trajectory(segs, sample_hz=sample_hz)
            self.last_traj = traj

            ok, constraint_msgs = evaluate_constraints(segs)
            cost = jerk_cost(traj)

            t0 = float(work_wps[0].t)
            t1 = float(work_wps[-1].t)

            # Objective label
            obj_labels = []
            if min_time:
                obj_labels.append("min time")
            if min_jerk:
                obj_labels.append("min jerk^2")
            obj_str = ", ".join(obj_labels) if obj_labels else "(none)"

            status_lines = [
                f"Subdivision: {n_sub}x (per original segment)" if n_sub>1 else "Subdivision: 1x",
                f"Segments: {len(segs)}  |  Type: {kind}  |  Samples: {traj.shape[0]}",
                f"Total time: {t1 - t0:.6f} s",
                f"Jerk cost (approx ∫||j||² dt): {cost:.6g}",
                f"Objectives: {obj_str}",
            ]

            if opt_info is not None:
                if "objective" in opt_info:
                    status_lines.append(f"Optimizer objective: {opt_info['objective']:.6g}")
                status_lines.append(
                    f"Optimizer: success={opt_info.get('success', False)} nit={opt_info.get('nit', 0)} msg={opt_info.get('message', '')}"
                )
                if "T_total" in opt_info:
                    status_lines.append(f"  Optimized total time: {opt_info['T_total']:.6f} s")
                if "env_ok" in opt_info:
                    status_lines.append(f"  env_ok: {opt_info['env_ok']}")
                if "jerk_ok" in opt_info:
                    status_lines.append(f"  jerk_ok: {opt_info['jerk_ok']}")
                if "acc_ok" in opt_info:
                    status_lines.append(f"  acc_ok: {opt_info['acc_ok']}")



            if constraint_msgs:
                status_lines += [""] + constraint_msgs
            else:
                status_lines.append("Constraints: OK")

            # Helpful heads-up: min-time without any dynamic constraint will just hit min_seg_dt
            if min_time and (not jerk_on):
                status_lines.append(
                    "Note: With jerk limit disabled, min-time will tend to drive segment times to Min segment dt. "
                    "Once you add dynamic limits (jerk/accel/torque), min-time becomes meaningful."
                )

            self.status.setText("\n".join(status_lines))

            # --- Plot ---
            env_for_plot = env if env is not None else self.current_envelope()
            self.update_plots(traj, env_for_plot, show_env=(env is not None))

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    def update_plots(self, traj: np.ndarray, env: WorkEnvelope, show_env: bool):
        t = traj[:, 0]
        x, y, z = traj[:, 1], traj[:, 2], traj[:, 3]
        vx, vy, vz = traj[:, 4], traj[:, 5], traj[:, 6]
        ax, ay, az = traj[:, 7], traj[:, 8], traj[:, 9]
        jx, jy, jz = traj[:, 10], traj[:, 11], traj[:, 12]
        jmag = np.sqrt(jx * jx + jy * jy + jz * jz)

        # clear
        self.plot_p.clear()
        self.plot_v.clear()
        self.plot_a.clear()
        self.plot_j.clear()

        # ----------------
        # Position
        # ----------------
        self.plot_p.addLegend()
        self.plot_p.plot(t, x, name="x")
        self.plot_p.plot(t, y, name="y")
        self.plot_p.plot(t, z, name="z")
        self.plot_p.showGrid(x=True, y=True)
        if show_env:
            R = max(0.0, env.cyl_radius - env.margin)
            zmin = env.z_min + env.margin
            zmax = env.z_max - env.margin
            self.plot_p.addLine(y=R, pen=pg.mkPen(style=Qt.PenStyle.DashLine))
            self.plot_p.addLine(y=zmin, pen=pg.mkPen(style=Qt.PenStyle.DashLine))
            self.plot_p.addLine(y=zmax, pen=pg.mkPen(style=Qt.PenStyle.DashLine))
        self.plot_p.setLabel("bottom", "time", units="s")
        self.plot_p.setLabel("left", "pos", units="m")

        # ----------------
        # Velocity
        # ----------------
        self.plot_v.addLegend()
        self.plot_v.plot(t, vx, name="vx")
        self.plot_v.plot(t, vy, name="vy")
        self.plot_v.plot(t, vz, name="vz")
        self.plot_v.showGrid(x=True, y=True)
        self.plot_v.setLabel("bottom", "time", units="s")
        self.plot_v.setLabel("left", "vel", units="m/s")

        # ----------------
        # Acceleration
        # ----------------
        self.plot_a.addLegend()
        self.plot_a.plot(t, ax, name="ax")
        self.plot_a.plot(t, ay, name="ay")
        self.plot_a.plot(t, az, name="az")
        self.plot_a.showGrid(x=True, y=True)
        self.plot_a.setLabel("bottom", "time", units="s")
        self.plot_a.setLabel("left", "acc", units="m/s^2")

        # ----------------
        # Jerk
        # ----------------
        self.plot_j.addLegend()
        self.plot_j.plot(t, jmag, name="||j||")
        self.plot_j.plot(t, jx, name="jx")
        self.plot_j.plot(t, jy, name="jy")
        self.plot_j.plot(t, jz, name="jz")
        self.plot_j.showGrid(x=True, y=True)
        self.plot_j.setLabel("bottom", "time", units="s")
        self.plot_j.setLabel("left", "jerk", units="m/s^3")

    def export_csv(self):
        if self.last_traj is None:
            QMessageBox.information(self, "Export", "No trajectory yet. Click 'Rebuild & Plot' first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save trajectory CSV", "traj.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            header = ["t","x","y","z","vx","vy","vz","ax","ay","az","jx","jy","jz"]
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for row in self.last_traj:
                    w.writerow([f"{v:.6f}" for v in row])
            QMessageBox.information(self, "Export", f"Wrote {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def reset_waypoints(self):
        self.waypoints = self.default_waypoints()
        self.populate_table()
        self.rebuild_and_plot()

    def add_waypoint(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        # default: append after last time + 0.2
        last_t = float(self.table.item(r-1, 0).text()) if r > 0 and self.table.item(r-1, 0) else 0.0
        self._set_cell(r, 0, last_t + 0.2)
        for c in range(1, 10):
            self._set_cell(r, c, 0.0)
        self.table.resizeColumnsToContents()

    def delete_selected(self):
        rows = sorted(set([idx.row() for idx in self.table.selectedIndexes()]), reverse=True)
        for r in rows:
            self.table.removeRow(r)


def main():
    app = QApplication(sys.argv)
    w = PathPlayground()
    w.resize(1200, 750)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
