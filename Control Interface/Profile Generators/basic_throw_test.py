import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from typing import Optional


# ----------------------------
# Cubic helpers
# ----------------------------
@dataclass
class CubicSeg:
    p0: np.ndarray  # (3,) position at start
    v0: np.ndarray  # (3,) velocity at start
    p1: np.ndarray  # (3,) position at end
    v1: np.ndarray  # (3,) velocity at end
    T: float        # duration (s)


def eval_cubic_hermite(p0, v0, p1, v1, T, t):
    """
    1D cubic Hermite.
    Returns (p, v, a, j) at local time t in [0,T].
    Jerk j is constant over the segment.
    """
    if T <= 0:
        raise ValueError("T must be > 0")
    s = max(0.0, min(1.0, t / T))

    # Hermite basis
    h00 =  2*s**3 - 3*s**2 + 1
    h10 =    s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 =    s**3 -   s**2

    p = h00*p0 + h10*(T*v0) + h01*p1 + h11*(T*v1)

    # first derivative wrt s
    dh00 =  6*s**2 - 6*s
    dh10 =  3*s**2 - 4*s + 1
    dh01 = -6*s**2 + 6*s
    dh11 =  3*s**2 - 2*s
    v = (dh00*p0 + dh10*(T*v0) + dh01*p1 + dh11*(T*v1)) / T

    # second derivative wrt s
    d2h00 = 12*s - 6
    d2h10 =  6*s - 4
    d2h01 = -12*s + 6
    d2h11 =  6*s - 2
    a = (d2h00*p0 + d2h10*(T*v0) + d2h01*p1 + d2h11*(T*v1)) / (T*T)

    # constant jerk over the segment
    j = (12*(p0 - p1) + 6*T*(v0 + v1)) / (T**3)

    return p, v, a, j


def sample_cubic_segment(seg: CubicSeg, t0: float, dt: float) -> np.ndarray:
    """
    Sample a 3D cubic segment into rows:
      [t, x,y,z, vx,vy,vz, ax,ay,az, jx,jy,jz]
    """
    T = float(seg.T)
    n = int(math.floor(T / dt))
    ts = [k*dt for k in range(n)]
    if (len(ts) == 0) or (abs(ts[-1] - T) > 1e-12):
        ts.append(T)

    out = []
    for tl in ts:
        p = []; v = []; a = []; j = []
        for axis in range(3):
            pi, vi, ai, ji = eval_cubic_hermite(
                seg.p0[axis], seg.v0[axis], seg.p1[axis], seg.v1[axis], seg.T, tl
            )
            p.append(pi); v.append(vi); a.append(ai); j.append(ji)
        out.append([t0 + tl] + p + v + a + j)

    return np.array(out, dtype=float)

def cubic_jerk(seg: CubicSeg):
    # per axis constant jerk
    j = np.zeros(3)
    for k in range(3):
        p0, v0, p1, v1, T = seg.p0[k], seg.v0[k], seg.p1[k], seg.v1[k], seg.T
        j[k] = (12 * (p0 - p1) + 6 * T * (v0 + v1)) / (T ** 3)
    return j


# ----------------------------
# Quintic helpers
# ----------------------------
def quintic_coeffs(p0, v0, a0, p1, v1, a1, T) -> np.ndarray:
    """
    Quintic polynomial coefficients for position:
      p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5,  t in [0,T]
    Constraints: p,v,a at t=0 and t=T.
    """
    T2, T3, T4, T5 = T*T, T*T*T, T*T*T*T, T*T*T*T*T
    c0 = p0
    c1 = v0
    c2 = 0.5 * a0

    # Solve for c3,c4,c5
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


def eval_quintic(c: np.ndarray, t: float) -> Tuple[float, float, float, float]:
    """Return position, velocity, acceleration, jerk at time t."""
    c0, c1, c2, c3, c4, c5 = c
    p = c0 + c1*t + c2*t*t + c3*t**3 + c4*t**4 + c5*t**5
    v = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    a = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
    j = 6*c3 + 24*c4*t + 60*c5*t**2
    return p, v, a, j


def sample_quintic_segment(c_xyz: np.ndarray, T: float, t0: float, dt: float) -> np.ndarray:
    """
    Sample a 3D quintic segment.
    Returns rows: [t, x,y,z, vx,vy,vz, ax,ay,az, jx,jy,jz]
    """
    n = int(math.floor(T / dt))
    ts = [k*dt for k in range(n)]
    if (len(ts) == 0) or (abs(ts[-1] - T) > 1e-12):
        ts.append(T)

    out = []
    for tl in ts:
        row = [t0 + tl]
        p = []
        v = []
        a = []
        j = []
        for axis in range(3):
            pi, vi, ai, ji = eval_quintic(c_xyz[axis], tl)
            p.append(pi); v.append(vi); a.append(ai); j.append(ji)
        row += p + v + a + j
        out.append(row)
    return np.array(out, dtype=float)

@dataclass
class WorkEnvelope:
    z_min: float
    z_max: float
    cyl_radius: float  # meters
    margin: float = 0.0  # optional conservative shrink (m)

def collocation_alphas(n: int, chebyshev: bool = True) -> np.ndarray:
    """Fractions in [0,1] for collocation points."""
    n = int(max(2, n))
    if chebyshev:
        k = np.arange(n)
        return 0.5 * (1.0 - np.cos(np.pi * k / (n - 1)))
    else:
        return np.linspace(0.0, 1.0, n)

def check_segment_envelope(c_xyz: np.ndarray, T: float, env: WorkEnvelope,
                           n_col: int = 10, chebyshev: bool = True):
    """
    Enforce cylinder envelope at collocation points for a 3D quintic segment.
    Returns: (ok, worst_dict)
      worst_dict has keys: 'zmin','zmax','rad' with (margin, t_local, xyz)
      margins are >=0 when satisfied.
    """
    R = max(0.0, env.cyl_radius - env.margin)
    zmin = env.z_min + env.margin
    zmax = env.z_max - env.margin

    worst = {
        "zmin": (+np.inf, None, None),
        "zmax": (+np.inf, None, None),
        "rad":  (+np.inf, None, None),
    }

    for a in collocation_alphas(n_col, chebyshev=chebyshev):
        tl = float(a) * float(T)

        x, _, _, _ = eval_quintic(c_xyz[0], tl)
        y, _, _, _ = eval_quintic(c_xyz[1], tl)
        z, _, _, _ = eval_quintic(c_xyz[2], tl)

        m_zmin = z - zmin
        m_zmax = zmax - z
        m_rad  = (R * R) - (x * x + y * y)

        if m_zmin < worst["zmin"][0]:
            worst["zmin"] = (float(m_zmin), tl, (float(x), float(y), float(z)))
        if m_zmax < worst["zmax"][0]:
            worst["zmax"] = (float(m_zmax), tl, (float(x), float(y), float(z)))
        if m_rad < worst["rad"][0]:
            worst["rad"]  = (float(m_rad),  tl, (float(x), float(y), float(z)))

    ok = (worst["zmin"][0] >= 0.0) and (worst["zmax"][0] >= 0.0) and (worst["rad"][0] >= 0.0)
    return ok, worst

def check_cubic_segment_envelope(seg: CubicSeg, env: WorkEnvelope,
                                 n_col: int = 10, chebyshev: bool = True):
    """
    Enforce cylinder envelope at collocation points for a 3D cubic segment.
    Returns: (ok, worst_dict) where margins must be >= 0.
    """
    R = max(0.0, env.cyl_radius - env.margin)
    zmin = env.z_min + env.margin
    zmax = env.z_max - env.margin

    worst = {
        "zmin": (+np.inf, None, None),
        "zmax": (+np.inf, None, None),
        "rad":  (+np.inf, None, None),
    }

    for a in collocation_alphas(n_col, chebyshev=chebyshev):
        tl = float(a) * float(seg.T)

        x, _, _, _ = eval_cubic_hermite(seg.p0[0], seg.v0[0], seg.p1[0], seg.v1[0], seg.T, tl)
        y, _, _, _ = eval_cubic_hermite(seg.p0[1], seg.v0[1], seg.p1[1], seg.v1[1], seg.T, tl)
        z, _, _, _ = eval_cubic_hermite(seg.p0[2], seg.v0[2], seg.p1[2], seg.v1[2], seg.T, tl)

        m_zmin = z - zmin
        m_zmax = zmax - z
        m_rad  = (R * R) - (x * x + y * y)

        if m_zmin < worst["zmin"][0]:
            worst["zmin"] = (float(m_zmin), tl, (float(x), float(y), float(z)))
        if m_zmax < worst["zmax"][0]:
            worst["zmax"] = (float(m_zmax), tl, (float(x), float(y), float(z)))
        if m_rad < worst["rad"][0]:
            worst["rad"]  = (float(m_rad),  tl, (float(x), float(y), float(z)))

    ok = (worst["zmin"][0] >= 0.0) and (worst["zmax"][0] >= 0.0) and (worst["rad"][0] >= 0.0)
    return ok, worst



def format_worst(name: str, worst_entry):
    m, tl, xyz = worst_entry
    if tl is None:
        return f"{name}: n/a"
    x, y, z = xyz
    return f"{name}: margin={m:+.6f} at t_local={tl:.4f}s  xyz=({x:+.4f},{y:+.4f},{z:+.4f})"

def compute_T_for_downward_accel_limit_cubic(D: float, a_down_max: float) -> float:
    """
    For cubic Hermite with v0=v1=0 moving displacement D,
      peak acceleration magnitude is |a|max = 6|D| / T^2.
    Solve for T such that |a|max <= a_down_max.
    """
    return math.sqrt((6.0 * abs(D)) / max(1e-9, a_down_max))


# ----------------------------
# Outputs
# ----------------------------
def write_traj_csv(path: str, traj: np.ndarray) -> None:
    """
    Writes: t, x,y,z, vx,vy,vz, ax,ay,az, jx,jy,jz
    Units: SI (m, m/s, m/s^2, m/s^3)
    """
    header = ["t","x","y","z","vx","vy","vz","ax","ay","az","jx","jy","jz"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in traj:
            w.writerow([f"{v:.6f}" for v in r])


def write_pose_cmd_csv(path: str, traj: np.ndarray,
                       roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> None:
    """
    Pose command format:
      t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"])
        for r in traj:
            t = float(r[0])
            x_mm = float(r[1]) * 1000.0
            y_mm = float(r[2]) * 1000.0
            z_mm = float(r[3]) * 1000.0
            w.writerow([f"{t:.6f}", f"{x_mm:.6f}", f"{y_mm:.6f}", f"{z_mm:.6f}",
                        f"{roll_deg:.6f}", f"{pitch_deg:.6f}", f"{yaw_deg:.6f}"])


# ----------------------------
# Main profile builder
# ----------------------------
def build_basic_throw_profile(
    z_bottom: float,
    T_air: float,
    sample_hz: float,
    g_mag: float,
    a_down_limit_g: float,
    T_hold_start: float,
    T_hold_end: float,
    T_down: Optional[float],
    T_up: float,
    T_return: float,
    envelope: Optional[WorkEnvelope] = None,
    n_col: int = 10,
    auto_retime: bool = False,
    retime_scale: float = 1.10,
    retime_max_iters: int = 30,
    jerk_limit: Optional[float] = None,

) -> Tuple[np.ndarray, dict]:
    """
    Steps:
    1) (optional) hold at 0,0,0
    2) move down to (0,0,z_bottom) with downward accel limit (via choosing T_down if not provided)
    3) move up to (0,0,0) and hit release velocity vz_release = g*T_air/2 at the end (hard)
    4) return to (0,0,0) with v=0 (optional shaping segment), then hold at origin

    Note: During flight we keep the hand at origin (after the return segment + hold).
    The "throw" happens at the end of the up-segment when we cross the origin with vz_release.
    """
    dt = 1.0 / sample_hz
    g = abs(g_mag)

    # Release velocity required for a ball to leave and return to z=0 after T_air
    vz_release = 0.5 * g * T_air

    # Choose T_down if not specified using accel limit in Z.
    # Your requirement: constrain to -0.5g acceleration in z direction during the down move.
    # We interpret this as: do not exceed downward acceleration magnitude a_down_max = 0.5g.
    a_down_max = a_down_limit_g * g
    if T_down is None:
        T_down = compute_T_for_downward_accel_limit_cubic(D=z_bottom - 0.0, a_down_max=a_down_max)
        # add a little margin so we don't sit exactly at the limit
        T_down *= 1.10

    def ensure_env_for_segment(name, make_seg_fn, T_initial):
        Tcur = float(T_initial)

        def ok_all(seg):
            # jerk check first (if enabled)
            if jerk_limit is not None:
                j = cubic_jerk(seg)
                if np.any(np.abs(j) > jerk_limit):
                    return False, {"jerk": j}

            # envelope check (if enabled)
            if envelope is None:
                return True, None
            ok_env, worst = check_cubic_segment_envelope(seg, envelope, n_col=n_col, chebyshev=True)
            if not ok_env:
                return False, worst
            return True, None

        seg = make_seg_fn(Tcur)
        ok, info_bad = ok_all(seg)
        if ok:
            return Tcur, seg, True, None

        if not auto_retime:
            return Tcur, seg, False, info_bad

        for _ in range(retime_max_iters):
            Tcur *= float(retime_scale)
            seg = make_seg_fn(Tcur)
            ok, info_bad = ok_all(seg)
            if ok:
                return Tcur, seg, True, None

        return Tcur, seg, False, info_bad

    # Segment A: hold at origin
    rows = []
    t0 = 0.0
    if T_hold_start > 0:
        n = int(math.floor(T_hold_start / dt))
        for k in range(n):
            rows.append([t0 + k*dt, 0,0,0, 0,0,0, 0,0,0, 0,0,0])
        t0 += T_hold_start

    # Segment B: origin -> bottom, v=0 at ends (cubic Hermite)
    p_origin = np.array([0.0, 0.0, 0.0], dtype=float)
    v_zero = np.array([0.0, 0.0, 0.0], dtype=float)
    p_bottom = np.array([0.0, 0.0, z_bottom], dtype=float)

    def make_B(Tseg):
        return CubicSeg(p0=p_origin, v0=v_zero, p1=p_bottom, v1=v_zero, T=float(Tseg))

    T_down, segB, okB, worstB = ensure_env_for_segment("B", make_B, T_down)
    if not okB:
        if isinstance(worstB, dict) and ("jerk" in worstB):
            j = worstB["jerk"]
            raise ValueError(f"Jerk limit violation in segment B: j={j} limit={jerk_limit}")
        if envelope is not None:
            raise ValueError("Envelope violation in segment B (origin->bottom):\n"
                             + format_worst("zmin", worstB["zmin"]) + "\n"
                             + format_worst("zmax", worstB["zmax"]) + "\n"
                             + format_worst("rad", worstB["rad"]))
        raise ValueError("Segment B failed for unknown reason.")

    rows.append(sample_cubic_segment(segB, t0, dt))
    t0 += T_down

    # Segment C: bottom -> origin (release), with end velocity = v_release
    v_release = np.array([0.0, 0.0, vz_release], dtype=float)

    def make_C(Tseg):
        return CubicSeg(p0=p_bottom, v0=v_zero, p1=p_origin, v1=v_release, T=float(Tseg))

    T_up, segC, okC, worstC = ensure_env_for_segment("C", make_C, T_up)
    if not okC:
        if isinstance(worstC, dict) and ("jerk" in worstC):
            j = worstC["jerk"]
            raise ValueError(f"Jerk limit violation in segment C: j={j} limit={jerk_limit}")
        if envelope is not None:
            raise ValueError("Envelope violation in segment C (origin->bottom):\n"
                             + format_worst("zmin", worstC["zmin"]) + "\n"
                             + format_worst("zmax", worstC["zmax"]) + "\n"
                             + format_worst("rad", worstC["rad"]))
        raise ValueError("Segment C failed for unknown reason.")

    rows.append(sample_cubic_segment(segC, t0, dt))
    t_release = t0 + T_up
    t0 += T_up

    # Segment D: settle at origin, end velocity 0
    def make_D(Tseg):
        return CubicSeg(p0=p_origin, v0=v_release, p1=p_origin, v1=v_zero, T=float(Tseg))

    T_return, segD, okD, worstD = ensure_env_for_segment("D", make_D, T_return)
    if not okD:
        if isinstance(worstD, dict) and ("jerk" in worstD):
            j = worstD["jerk"]
            raise ValueError(f"Jerk limit violation in segment D: j={j} limit={jerk_limit}")
        if envelope is not None:
            raise ValueError("Envelope violation in segment D (origin->bottom):\n"
                             + format_worst("zmin", worstD["zmin"]) + "\n"
                             + format_worst("zmax", worstD["zmax"]) + "\n"
                             + format_worst("rad", worstD["rad"]))
        raise ValueError("Segment D failed for unknown reason.")

    rows.append(sample_cubic_segment(segD, t0, dt))
    t0 += T_return

    # Segment E: hold at origin until at least catch time, then additional hold end
    # Ball catch time is t_release + T_air. Ensure we hold through that.
    t_catch = t_release + T_air
    hold_until = max(t0, t_catch)
    if hold_until > t0:
        n = int(math.ceil((hold_until - t0) / dt))
        for k in range(n):
            rows.append([t0 + k*dt, 0,0,0, 0,0,0, 0,0,0, 0,0,0])
        t0 = hold_until

    if T_hold_end > 0:
        n = int(math.floor(T_hold_end / dt))
        for k in range(n):
            rows.append([t0 + k*dt, 0,0,0, 0,0,0, 0,0,0, 0,0,0])
        t0 += T_hold_end

    # Concatenate
    traj = np.vstack([r if isinstance(r, np.ndarray) else np.array([r], dtype=float) for r in rows])

    # Diagnostics: check downward accel limit during segment B
    # (We only check Z accel; negative is downward.)
    az = traj[:, 9]  # az column
    az_min = float(np.min(az))
    info = {
        "vz_release": float(vz_release),
        "t_release": float(t_release),
        "t_catch": float(t_catch),
        "T_down": float(T_down),
        "az_min": az_min,
        "a_down_max": float(-a_down_max),  # negative direction
    }
    return traj, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-bottom", type=float, default=-0.20,
                    help="Bottom-of-stroke Z [m] (negative is down).")
    ap.add_argument("--t-air", type=float, default=1.0,
                    help="Ball air time from release at z=0 to catch at z=0 [s].")
    ap.add_argument("--g", type=float, default=9.81, help="Gravity magnitude [m/s^2].")

    ap.add_argument("--a-down-limit-g", type=float, default=0.5,
                    help="Max downward hand acceleration magnitude during down move, in g's (e.g. 0.5 -> -0.5g limit).")
    ap.add_argument("--t-down", type=float, default=None,
                    help="Optional explicit down-move duration [s]. If omitted, computed from accel limit.")
    ap.add_argument("--t-up", type=float, default=0.35,
                    help="Duration from bottom to release at origin [s].")
    ap.add_argument("--t-return", type=float, default=0.20,
                    help="Duration to settle back to origin with v=0 after release [s].")

    ap.add_argument("--hold-start", type=float, default=0.0,
                    help="Optional hold time at origin before moving down [s].")
    ap.add_argument("--hold-end", type=float, default=1.0,
                    help="Hold time at origin after catch time [s].")

    ap.add_argument("--sample-hz", type=float, default=200.0)
    ap.add_argument("--traj-out", type=str, default="traj.csv")
    ap.add_argument("--pose-out", type=str, default="pose_cmd.csv")

    ap.add_argument("--z-min", type=float, default=-0.30, help="Work envelope z_min [m]")
    ap.add_argument("--z-max", type=float, default=+0.30, help="Work envelope z_max [m]")
    ap.add_argument("--cyl-radius", type=float, default=0.25, help="Work envelope cylinder radius [m]")
    ap.add_argument("--env-margin", type=float, default=0.00, help="Conservative margin [m]")
    ap.add_argument("--collocation", type=int, default=10, help="Collocation points per segment")
    ap.add_argument("--auto-retime", action="store_true", help="If set, increase segment times until envelope is satisfied")
    ap.add_argument("--retime-scale", type=float, default=1.10, help="Time scaling factor per iteration")
    ap.add_argument("--retime-max-iters", type=int, default=30, help="Max retime iterations per segment")

    ap.add_argument("--jerk-limit", type=float, default=None,
                    help="Optional jerk limit magnitude [m/s^3]. If set, checks |j| per segment (XYZ).")
    ap.add_argument("--jerk-limit-roll", type=float, default=None,
                    help="Optional jerk limit for roll/pitch later [rad/s^3].")

    args = ap.parse_args()

    env = WorkEnvelope(
        z_min=args.z_min,
        z_max=args.z_max,
        cyl_radius=args.cyl_radius,
        margin=args.env_margin,
    )

    traj, info = build_basic_throw_profile(
        z_bottom=args.z_bottom,
        T_air=args.t_air,
        sample_hz=args.sample_hz,
        g_mag=args.g,
        a_down_limit_g=args.a_down_limit_g,
        T_hold_start=args.hold_start,
        T_hold_end=args.hold_end,
        T_down=args.t_down,
        T_up=args.t_up,
        T_return=args.t_return,
        envelope=env,
        n_col=args.collocation,
        auto_retime=args.auto_retime,
        retime_scale=args.retime_scale,
        retime_max_iters=args.retime_max_iters,
        jerk_limit=args.jerk_limit,
    )

    write_traj_csv(args.traj_out, traj)
    write_pose_cmd_csv(args.pose_out, traj, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)

    # Summary
    v = np.linalg.norm(traj[:, 4:7], axis=1)
    a = np.linalg.norm(traj[:, 7:10], axis=1)
    print("Basic throw test profile generated")
    print(f"  z_bottom: {args.z_bottom:.3f} m")
    print(f"  t_air: {args.t_air:.3f} s -> vz_release (hard): {info['vz_release']:.3f} m/s")
    print(f"  computed t_down: {info['T_down']:.3f} s (or user-specified)")
    print(f"  min az in traj: {info['az_min']:.3f} m/s^2 (limit is >= {-abs(args.a_down_limit_g*args.g):.3f})")
    print(f"  release time: {info['t_release']:.3f} s, catch time: {info['t_catch']:.3f} s")
    print(f"  max speed: {float(np.max(v)):.3f} m/s, max accel: {float(np.max(a)):.3f} m/s^2")
    print(f"Wrote: {args.traj_out}, {args.pose_out}")


if __name__ == "__main__":
    main()
