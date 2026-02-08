# jugglepath.py
# ============================================================
# Waypoint + Segment based path authoring for juggling motions
#
# Key semantics:
#  - Waypoints define constraints (p, optional v/a/t)
#  - Segments define how to get to the next waypoint
#  - s_curve_monotonic = LineDVNoCoastScaled (existing primitive)
#  - s_curve = generic accel+decel S-curve (not implemented yet)
#
# Defaults:
#  - v = 0 unless waypoint defines it
#  - a = 0 unless waypoint defines it
#
# ============================================================

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal
import numpy as np
import math

# ------------------------------------------------------------
# Core state / result containers
# ------------------------------------------------------------

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
    traj: np.ndarray          # (N,13): t,x,y,z,vx,vy,vz,ax,ay,az,jx,jy,jz
    end_state: State3D
    info: Dict[str, object]


@dataclass
class PathResult:
    traj: np.ndarray
    segment_infos: List[Dict[str, object]]
    end_state: State3D


# ------------------------------------------------------------
# Waypoints and segment specs
# ------------------------------------------------------------

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
    time_law: Literal["linear", "s_curve_monotonic", "s_curve"] = "linear"
    duration: Optional[float] = None

    accel_ref: float = 1.0
    jerk_ref: float = 10.0


# ------------------------------------------------------------
# Primitive base class
# ------------------------------------------------------------

class Primitive3D:
    def generate(self, start: State3D, sample_hz: float) -> SegmentResult:
        raise NotImplementedError


# ------------------------------------------------------------
# Linear time-law line primitive
# ------------------------------------------------------------

class LineLinear(Primitive3D):
    def __init__(self, p1, duration: Optional[float] = None, nominal_speed: float = 0.5):
        self.p1 = np.asarray(p1, dtype=float).reshape(3)
        self.duration = duration
        self.nominal_speed = float(nominal_speed)

    def generate(self, start: State3D, sample_hz: float) -> SegmentResult:
        dt = 1.0 / sample_hz
        p0 = start.p
        dp = self.p1 - p0
        L = float(np.linalg.norm(dp))

        if L < 1e-12:
            traj = np.zeros((1, 13))
            traj[0, 1:4] = p0
            traj[0, 4:7] = start.v
            traj[0, 7:10] = start.a
            return SegmentResult(traj, start, {"mode": "line_linear", "degenerate": True})

        u = dp / L

        if self.duration is not None:
            T = max(1e-9, self.duration)
        else:
            T = L / max(1e-9, self.nominal_speed)

        n = max(2, int(math.ceil(T / dt)) + 1)
        t = np.linspace(0.0, T, n)
        s = (L / T) * t

        p = p0[None, :] + s[:, None] * u
        v = np.full((n, 1), L / T) * u
        a = np.zeros_like(p)
        j = np.zeros_like(p)

        traj = np.zeros((n, 13))
        traj[:, 0] = t
        traj[:, 1:4] = p
        traj[:, 4:7] = v
        traj[:, 7:10] = a
        traj[:, 10:13] = j

        end = State3D(p[-1], v[-1], a[-1])
        return SegmentResult(traj, end, {
            "mode": "line_linear",
            "L": L,
            "t_total": T,
        })


# ------------------------------------------------------------
# Existing monotonic S-curve primitive (assumed to exist)
# ------------------------------------------------------------

class LineDVNoCoastScaled(Primitive3D):
    """
    Existing implementation:
      - monotonic accel/decel
      - dv-constrained
      - jerk/accel scaled to hit distance
    """
    def __init__(self, p1, v1_along, accel_ref, jerk_ref,
                 scale_accel=True, scale_jerk=True):
        self.p1 = np.asarray(p1, dtype=float).reshape(3)
        self.v1_along = float(v1_along)
        self.accel_ref = float(accel_ref)
        self.jerk_ref = float(jerk_ref)
        self.scale_accel = scale_accel
        self.scale_jerk = scale_jerk

    def generate(self, start: State3D, sample_hz: float) -> SegmentResult:
        raise NotImplementedError(
            "LineDVNoCoastScaled implementation assumed to exist elsewhere"
        )


# ------------------------------------------------------------
# JugglePath: authoritative container
# ------------------------------------------------------------

class JugglePath:
    def __init__(self, sample_hz: float, start: Optional[State3D] = None):
        self.sample_hz = float(sample_hz)

        if start is None:
            start = State3D(
                p=np.zeros(3),
                v=np.zeros(3),
                a=np.zeros(3),
            )

        self.waypoints: List[Waypoint] = [
            Waypoint(p=start.p, v=start.v, a=start.a, t=0.0)
        ]
        self.segments: List[SegmentSpec] = []

    # -----------------------------
    # Authoring API
    # -----------------------------

    def add_segment(
        self,
        p,
        v=None,
        a=None,
        t=None,
        *,
        curve="line",
        time_law="linear",
        duration=None,
        accel_ref=1.0,
        jerk_ref=10.0,
    ):
        self.segments.append(SegmentSpec(
            curve=curve,
            time_law=time_law,
            duration=duration,
            accel_ref=accel_ref,
            jerk_ref=jerk_ref,
        ))
        self.waypoints.append(Waypoint(p=p, v=v, a=a, t=t))
        return self

    def set_waypoint(self, i, *, p=None, v=None, a=None, t=None):
        wp = self.waypoints[i]
        if p is not None: wp.p = np.asarray(p, dtype=float)
        if v is not None: wp.v = np.asarray(v, dtype=float)
        if a is not None: wp.a = np.asarray(a, dtype=float)
        if t is not None: wp.t = float(t)

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _segment_duration(self, i):
        seg = self.segments[i]
        if seg.duration is not None:
            return seg.duration

        w0 = self.waypoints[i]
        w1 = self.waypoints[i + 1]
        if w0.t is not None and w1.t is not None:
            T = w1.t - w0.t
            if T <= 0:
                raise ValueError("Waypoint times must be increasing")
            return T
        return None

    def _along_path_bc(self, wp, u):
        sd = float(np.dot(wp.v, u)) if wp.v is not None else 0.0
        sdd = float(np.dot(wp.a, u)) if wp.a is not None else 0.0
        return sd, sdd

    def _materialize_primitive(self, i, start_state: State3D):
        seg = self.segments[i]
        w0 = self.waypoints[i]
        w1 = self.waypoints[i + 1]

        dp = w1.p - start_state.p
        L = float(np.linalg.norm(dp))
        u = dp / L if L > 1e-12 else np.array([1.0, 0.0, 0.0])

        if seg.time_law == "linear":
            return LineLinear(
                p1=w1.p,
                duration=self._segment_duration(i),
            )

        if seg.time_law == "s_curve_monotonic":
            sd0, sdd0 = self._along_path_bc(w0, u)
            sd1, sdd1 = self._along_path_bc(w1, u)

            # NOTE:
            # accel defaults to 0 unless waypoint defines it.
            # LineDVNoCoastScaled does not enforce accel BCs yet,
            # but semantics are correct for future upgrade.
            return LineDVNoCoastScaled(
                p1=w1.p,
                v1_along=sd1,
                accel_ref=seg.accel_ref,
                jerk_ref=seg.jerk_ref,
                scale_accel=True,
                scale_jerk=True,
            )

        if seg.time_law == "s_curve":
            sd0, sdd0 = self._along_path_bc(w0, u)
            sd1, sdd1 = self._along_path_bc(w1, u)
            raise NotImplementedError(
                "Generic s_curve not implemented yet "
                "(accel+decel to satisfy end p,v,a)."
            )

        raise ValueError(f"Unknown time_law: {seg.time_law}")

    # -----------------------------
    # Build trajectory
    # -----------------------------

    def build(self) -> PathResult:
        if len(self.waypoints) < 2:
            empty = np.zeros((0, 13))
            return PathResult(empty, [], State3D(np.zeros(3), np.zeros(3), np.zeros(3)))

        traj_all = []
        infos = []

        w0 = self.waypoints[0]
        cur = State3D(
            p=w0.p,
            v=w0.v if w0.v is not None else np.zeros(3),
            a=w0.a if w0.a is not None else np.zeros(3),
        )

        t_offset = 0.0

        for i in range(len(self.segments)):
            prim = self._materialize_primitive(i, cur)
            res = prim.generate(cur, self.sample_hz)

            traj = res.traj.copy()
            traj[:, 0] += t_offset

            if traj_all and traj.shape[0] > 0 and abs(traj[0, 0] - t_offset) < 1e-12:
                traj = traj[1:]

            if traj.shape[0] > 0:
                t_offset = traj[-1, 0]

            traj_all.append(traj)
            infos.append({
                "segment_index": i,
                "time_law": self.segments[i].time_law,
                **res.info,
            })

            cur = res.end_state

        full = np.vstack(traj_all) if traj_all else np.zeros((0, 13))
        return PathResult(full, infos, cur)
