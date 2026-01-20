# path_playground_gui_scurve_basic.py
# ------------------------------------------------------------
# S-curve-only task-space path playground (Z-only)
# Supports signed endpoint Z-velocities vz0/vz1 and |vz| <= v_max.
#
# Cases:
#   A) dz != 0:
#      Build normalized time law s(t) for s in [0,1] with endpoint sdot0/sdot1
#      then map to z(t)=z0+s(t)*(z1-z0)
#
#   B) dz == 0:
#      Generate an "excursion" trajectory that returns to z0 but changes velocity
#
# Plots: position, velocity, acceleration, jerk
# ------------------------------------------------------------

import sys
import math
import csv
from typing import Tuple, List

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QCheckBox,
    QFileDialog, QGroupBox, QMessageBox
)

import pyqtgraph as pg


# ------------------------------------------------------------
# Envelope model (Z-only for now)
# ------------------------------------------------------------
class WorkEnvelope:
    def __init__(self, z_min: float, z_max: float, margin: float = 0.0):
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.margin = float(margin)

    def contains_z(self, z: float) -> bool:
        return (self.z_min + self.margin) <= z <= (self.z_max - self.margin)


# ------------------------------------------------------------
# Helpers: exact integration under constant jerk
# State is (p, v, a)
# ------------------------------------------------------------
def _integrate_const_jerk(p: float, v: float, a: float, j: float, dt: float):
    """Exact integration for constant jerk over dt."""
    p1 = p + v * dt + 0.5 * a * dt * dt + (1.0 / 6.0) * j * dt**3
    v1 = v + a * dt + 0.5 * j * dt * dt
    a1 = a + j * dt
    return p1, v1, a1


def _build_min_time_dv_segments(v_start: float, v_end: float, amax: float, jmax: float) -> List[Tuple[float, float]]:
    """
    Jerk-limited, accel-limited, min-time velocity change from v_start to v_end,
    with a(t) starting/ending at 0.

    Returns list of (jerk, duration).

    Works for any real v_start/v_end (can be negative).
    """
    dv = v_end - v_start
    if abs(dv) < 1e-15:
        return []

    sgn = 1.0 if dv > 0 else -1.0
    dv = abs(dv)

    # Triangular accel candidate: a_peak = sqrt(dv * jmax)
    a_peak = math.sqrt(dv * jmax)

    if a_peak <= amax + 1e-15:
        Tj = a_peak / jmax
        return [(sgn * jmax, Tj), (-sgn * jmax, Tj)]

    # Trapezoidal accel (hits amax)
    Tj = amax / jmax
    # dv = amax * (Ta + Tj)
    Ta = dv / amax - Tj
    if Ta < 0:
        Ta = 0.0

    segs = [(sgn * jmax, Tj)]
    if Ta > 0:
        segs.append((0.0, Ta))
    segs.append((-sgn * jmax, Tj))
    return segs


def _simulate_segments(
    segments: List[Tuple[float, float]],
    p0: float,
    v0: float,
    a0: float,
    dt: float
) -> np.ndarray:
    """Sample rows [t, p, v, a, j] while integrating piecewise-constant jerk segments."""
    rows = []
    t = 0.0
    p = p0
    v = v0
    a = a0

    for (j, dur) in segments:
        if dur <= 0:
            continue
        t_end = t + dur
        while t < t_end - 1e-12:
            dt_step = min(dt, t_end - t)
            p, v, a = _integrate_const_jerk(p, v, a, j, dt_step)
            t += dt_step
            rows.append([t, p, v, a, j])

    return np.array(rows, dtype=float) if rows else np.zeros((0, 5), dtype=float)


def _phase_distance_time(v_start: float, v_end: float, amax: float, jmax: float) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Return (distance, time, segments) for min-time velocity-change phase."""
    segs = _build_min_time_dv_segments(v_start, v_end, amax, jmax)

    # exact integrate per segment boundary
    p = 0.0
    v = v_start
    a = 0.0
    t = 0.0
    for (j, dur) in segs:
        p, v, a = _integrate_const_jerk(p, v, a, j, dur)
        t += dur
    return p, t, segs


# ------------------------------------------------------------
# Normalized time-law planner: s(t) in [0,1] for dz != 0
# Inputs are endpoint speeds and limits in s-space.
# ------------------------------------------------------------
def generate_scurve_normalized_with_vbounds(
    sdot0: float,
    sdot1: float,
    sdot_max: float,
    amax: float,
    jmax: float,
    sample_hz: float
) -> Tuple[np.ndarray, dict]:
    """
    Build jerk-limited, accel-limited profile from s=0 to s=1 with endpoint velocities.

    Structure:
      sdot0 -> sdot_peak (min-time)
      optional cruise at sdot_peak
      sdot_peak -> sdot1 (min-time)

    Constraints:
      |sdot| <= sdot_max
      |sddot| <= amax
      |sdddot| <= jmax

    For simplicity, sdot_peak is chosen >= 0 (a forward peak). sdot0/sdot1 may be signed.
    Returns samples [t, s, sdot, sddot, sdddot].
    """
    if sample_hz <= 0:
        raise ValueError("sample_hz must be > 0")
    if amax <= 0 or jmax <= 0:
        raise ValueError("amax and jmax must be > 0")
    if sdot_max <= 0:
        raise ValueError("sdot_max must be > 0")

    sdot0 = float(sdot0)
    sdot1 = float(sdot1)
    sdot_max = float(sdot_max)

    if abs(sdot0) - sdot_max > 1e-9 or abs(sdot1) - sdot_max > 1e-9:
        raise ValueError("|vz0| or |vz1| exceed v_max (mapped to s-space)")

    # peak forward speed (>=0)
    v_lo = max(0.0, sdot0, sdot1)
    v_hi = sdot_max

    def d_min_for_peak(v_peak: float) -> float:
        d1, _, _ = _phase_distance_time(sdot0, v_peak, amax, jmax)
        d2, _, _ = _phase_distance_time(v_peak, sdot1, amax, jmax)
        return d1 + d2

    d_at_vhi = d_min_for_peak(v_hi)

    if d_at_vhi <= 1.0 + 1e-12:
        v_peak = v_hi
        d_min = d_at_vhi
        t_cruise = (1.0 - d_min) / v_peak if v_peak > 1e-15 else 0.0
        mode = "cruise_at_vmax" if t_cruise > 0 else "no_cruise_at_vmax"
    else:
        d_at_vlo = d_min_for_peak(v_lo)
        if d_at_vlo > 1.0 + 1e-12:
            raise ValueError(
                "Infeasible dz!=0 profile with given limits and endpoint speeds. "
                "Try increasing |z1-z0|, increasing amax/jmax, reducing |vz0|/|vz1|, or increasing v_max."
            )

        lo = v_lo
        hi = v_hi
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            dmid = d_min_for_peak(mid)
            if dmid > 1.0:
                hi = mid
            else:
                lo = mid
            if abs(dmid - 1.0) < 1e-12:
                break
        v_peak = lo
        t_cruise = 0.0
        mode = "limited_peak_no_cruise"

    segs1 = _build_min_time_dv_segments(sdot0, v_peak, amax, jmax)
    segs2 = _build_min_time_dv_segments(v_peak, sdot1, amax, jmax)

    segments: List[Tuple[float, float]] = []
    segments.extend(segs1)
    if t_cruise > 0:
        segments.append((0.0, t_cruise))
    segments.extend(segs2)

    dt = 1.0 / float(sample_hz)
    samples = _simulate_segments(segments, p0=0.0, v0=sdot0, a0=0.0, dt=dt)
    if samples.shape[0] == 0:
        raise ValueError("Degenerate profile")

    # Normalize so s_end == 1 exactly (keeps continuity; scales derivatives consistently)
    s_end = samples[-1, 1]
    if abs(s_end) < 1e-15:
        raise ValueError("Profile generation failed (near-zero distance)")
    scale = 1.0 / s_end
    samples[:, 1] *= scale
    samples[:, 2] *= scale
    samples[:, 3] *= scale
    samples[:, 4] *= scale

    info = {
        "mode": mode,
        "sdot_peak": float(v_peak),
        "t_total": float(samples[-1, 0]),
        "t_cruise": float(t_cruise),
    }
    return samples, info


def map_scurve_to_z(sc: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Map normalized samples [t, s, sdot, sddot, sjerk] to traj columns."""
    dz = z1 - z0
    traj = np.zeros((sc.shape[0], 13), dtype=float)
    traj[:, 0] = sc[:, 0]
    traj[:, 3] = z0 + sc[:, 1] * dz
    traj[:, 6] = sc[:, 2] * dz
    traj[:, 9] = sc[:, 3] * dz
    traj[:, 12] = sc[:, 4] * dz
    return traj


# ------------------------------------------------------------
# dz == 0 special case: return to same position while changing velocity
# ------------------------------------------------------------
def generate_z_profile_same_position(
    z0: float,
    vz0: float,
    vz1: float,
    vmax: float,
    amax_z: float,
    jmax_z: float,
    sample_hz: float,
) -> Tuple[np.ndarray, dict]:
    """
    Generate a Z trajectory that starts at z0 with vz0 and ends at z0 with vz1,
    under |vz|<=vmax, |az|<=amax_z, |jz|<=jmax_z.

    Structure:
      vz0 -> v_peak (min-time)
      optional cruise at v_peak
      v_peak -> vz1 (min-time)
    Solve v_peak and cruise such that net displacement == 0.
    """
    if sample_hz <= 0:
        raise ValueError("sample_hz must be > 0")
    if vmax <= 0 or amax_z <= 0 or jmax_z <= 0:
        raise ValueError("vmax, amax_z, jmax_z must be > 0")
    if abs(vz0) - vmax > 1e-9 or abs(vz1) - vmax > 1e-9:
        raise ValueError("|vz0| or |vz1| exceed v_max")

    vmin = -float(vmax)
    vmax = float(vmax)

    def d_min_for_peak(v_peak: float) -> float:
        d1, _, _ = _phase_distance_time(vz0, v_peak, amax_z, jmax_z)
        d2, _, _ = _phase_distance_time(v_peak, vz1, amax_z, jmax_z)
        return d1 + d2

    # Find a root (if any) for d_min_for_peak(v_peak) == 0 (no cruise)
    xs = np.linspace(vmin, vmax, 801)
    ds = np.array([d_min_for_peak(x) for x in xs], dtype=float)

    bracket = None
    for i in range(len(xs) - 1):
        if ds[i] == 0.0:
            bracket = (xs[i], xs[i])
            break
        if ds[i] * ds[i + 1] < 0:
            bracket = (xs[i], xs[i + 1])
            break

    if bracket is not None and bracket[0] != bracket[1]:
        lo, hi = float(bracket[0]), float(bracket[1])
        dlo = d_min_for_peak(lo)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            dmid = d_min_for_peak(mid)
            if abs(dmid) < 1e-12:
                lo = hi = mid
                break
            if dlo * dmid <= 0:
                hi = mid
            else:
                lo = mid
                dlo = dmid
        v_peak = 0.5 * (lo + hi)
        t_cruise = 0.0
        mode = "dz0_no_cruise"
    else:
        # Use cruise to cancel displacement: d_min + v_peak*t = 0
        # choose a nonzero bound v_peak that yields a nonnegative cruise time
        candidates = []
        for v_peak_try in (vmin, vmax):
            if abs(v_peak_try) < 1e-12:
                continue
            dmin = d_min_for_peak(v_peak_try)
            t = -dmin / v_peak_try
            if t >= -1e-12:  # allow tiny negative numeric
                candidates.append((max(0.0, t), v_peak_try, dmin))
        if not candidates:
            raise ValueError("Infeasible dz=0 profile with given limits; try increasing v_max/amax/jmax")
        t_cruise, v_peak, _ = min(candidates, key=lambda x: x[0])
        mode = "dz0_with_cruise"

    segs1 = _build_min_time_dv_segments(vz0, v_peak, amax_z, jmax_z)
    segs2 = _build_min_time_dv_segments(v_peak, vz1, amax_z, jmax_z)

    segments: List[Tuple[float, float]] = []
    segments.extend(segs1)
    if t_cruise > 0:
        segments.append((0.0, t_cruise))
    segments.extend(segs2)

    dt = 1.0 / float(sample_hz)
    samples = _simulate_segments(segments, p0=z0, v0=vz0, a0=0.0, dt=dt)
    if samples.shape[0] == 0:
        raise ValueError("Degenerate dz=0 profile")

    # tiny correction to land exactly at z0 (usually ~1e-15)
    z_end = samples[-1, 1]
    samples[:, 1] += (z0 - z_end)

    traj = np.zeros((samples.shape[0], 13), dtype=float)
    traj[:, 0] = samples[:, 0]
    traj[:, 3] = samples[:, 1]
    traj[:, 6] = samples[:, 2]
    traj[:, 9] = samples[:, 3]
    traj[:, 12] = samples[:, 4]

    info = {
        "mode": mode,
        "t_total": float(traj[-1, 0]),
        "v_peak": float(v_peak),
        "t_cruise": float(t_cruise),
    }
    return traj, info


# ------------------------------------------------------------
# GUI
# ------------------------------------------------------------
class ScurvePlayground(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z S-curve Playground (vz0/vz1/vmax)")

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 0)
        root.addLayout(right, 1)

        # ---- Move controls ----
        cg = QGroupBox("Move")
        left.addWidget(cg)
        g = QGridLayout(cg)

        row = 0
        g.addWidget(QLabel("z0 [m]"), row, 0)
        self.z0 = QDoubleSpinBox(); self.z0.setRange(-10, 10); self.z0.setDecimals(4); self.z0.setValue(0.0)
        g.addWidget(self.z0, row, 1); row += 1

        g.addWidget(QLabel("z1 [m]"), row, 0)
        self.z1 = QDoubleSpinBox(); self.z1.setRange(-10, 10); self.z1.setDecimals(4); self.z1.setValue(0.3)
        g.addWidget(self.z1, row, 1); row += 1

        g.addWidget(QLabel("vz0 [m/s]"), row, 0)
        self.vz0 = QDoubleSpinBox(); self.vz0.setRange(-100, 100); self.vz0.setDecimals(4); self.vz0.setValue(0.0)
        g.addWidget(self.vz0, row, 1); row += 1

        g.addWidget(QLabel("vz1 [m/s]"), row, 0)
        self.vz1 = QDoubleSpinBox(); self.vz1.setRange(-100, 100); self.vz1.setDecimals(4); self.vz1.setValue(0.0)
        g.addWidget(self.vz1, row, 1); row += 1

        g.addWidget(QLabel("v_max |vz| [m/s]"), row, 0)
        self.vmax = QDoubleSpinBox(); self.vmax.setRange(0.001, 1000); self.vmax.setDecimals(4); self.vmax.setValue(3.0)
        g.addWidget(self.vmax, row, 1); row += 1

        # ---- Limits ----
        lg = QGroupBox("Limits")
        left.addWidget(lg)
        lgd = QGridLayout(lg)

        lgd.addWidget(QLabel("amax_z [m/s²]"), 0, 0)
        self.amax = QDoubleSpinBox(); self.amax.setRange(0.01, 1e6); self.amax.setDecimals(3); self.amax.setValue(50.0)
        lgd.addWidget(self.amax, 0, 1)

        lgd.addWidget(QLabel("jmax_z [m/s³]"), 1, 0)
        self.jmax = QDoubleSpinBox(); self.jmax.setRange(0.01, 1e6); self.jmax.setDecimals(3); self.jmax.setValue(200.0)
        lgd.addWidget(self.jmax, 1, 1)

        lgd.addWidget(QLabel("Sample Hz"), 2, 0)
        self.sample_hz = QDoubleSpinBox(); self.sample_hz.setRange(10, 2000); self.sample_hz.setValue(200)
        lgd.addWidget(self.sample_hz, 2, 1)

        # ---- Envelope ----
        eg = QGroupBox("Envelope (Z)")
        left.addWidget(eg)
        egd = QGridLayout(eg)

        self.env_enable = QCheckBox("Enable envelope")
        self.env_enable.setChecked(True)
        egd.addWidget(self.env_enable, 0, 0, 1, 2)

        egd.addWidget(QLabel("z_min [m]"), 1, 0)
        self.zmin = QDoubleSpinBox(); self.zmin.setRange(-10, 10); self.zmin.setValue(-0.5)
        egd.addWidget(self.zmin, 1, 1)

        egd.addWidget(QLabel("z_max [m]"), 2, 0)
        self.zmax = QDoubleSpinBox(); self.zmax.setRange(-10, 10); self.zmax.setValue(0.5)
        egd.addWidget(self.zmax, 2, 1)

        egd.addWidget(QLabel("margin [m]"), 3, 0)
        self.margin = QDoubleSpinBox(); self.margin.setRange(0, 1); self.margin.setDecimals(4); self.margin.setValue(0.0)
        egd.addWidget(self.margin, 3, 1)

        # Buttons
        self.btn_run = QPushButton("Generate")
        self.btn_export = QPushButton("Export CSV")
        left.addWidget(self.btn_run)
        left.addWidget(self.btn_export)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        # Plots
        pg.setConfigOptions(antialias=True)
        self.plot_p = pg.PlotWidget(title="Position z")
        self.plot_v = pg.PlotWidget(title="Velocity vz")
        self.plot_a = pg.PlotWidget(title="Acceleration az")
        self.plot_j = pg.PlotWidget(title="Jerk jz")

        right.addWidget(self.plot_p, 1)
        right.addWidget(self.plot_v, 1)
        right.addWidget(self.plot_a, 1)
        right.addWidget(self.plot_j, 1)

        self.last_traj = None

        self.btn_run.clicked.connect(self.run)
        self.btn_export.clicked.connect(self.export_csv)

    def run(self):
        try:
            z0 = float(self.z0.value())
            z1 = float(self.z1.value())
            vz0 = float(self.vz0.value())
            vz1 = float(self.vz1.value())
            vmax = float(self.vmax.value())
            amax_z = float(self.amax.value())
            jmax_z = float(self.jmax.value())
            sample_hz = float(self.sample_hz.value())

            env = None
            if self.env_enable.isChecked():
                env = WorkEnvelope(self.zmin.value(), self.zmax.value(), self.margin.value())
                if not env.contains_z(z0) or not env.contains_z(z1):
                    raise ValueError("Z endpoints violate envelope")

            dz = z1 - z0
            D = abs(dz)

            if D < 1e-12:
                # dz == 0: change velocity while returning to same position
                traj, info = generate_z_profile_same_position(
                    z0=z0, vz0=vz0, vz1=vz1,
                    vmax=vmax, amax_z=amax_z, jmax_z=jmax_z,
                    sample_hz=sample_hz
                )
            else:
                # dz != 0: normalized s(t) then map to z
                # vz = sdot * dz  => sdot = vz / dz
                sdot0 = vz0 / dz
                sdot1 = vz1 / dz

                # bounds in s-space use magnitude scaling by |dz|
                sdot_max = abs(vmax) / D
                amax_s = amax_z / D
                jmax_s = jmax_z / D

                sc, info = generate_scurve_normalized_with_vbounds(
                    sdot0=sdot0, sdot1=sdot1,
                    sdot_max=sdot_max,
                    amax=amax_s, jmax=jmax_s,
                    sample_hz=sample_hz
                )
                traj = map_scurve_to_z(sc, z0, z1)

            # envelope check over trajectory
            if env is not None:
                if np.any(traj[:, 3] < env.z_min + env.margin) or np.any(traj[:, 3] > env.z_max - env.margin):
                    raise ValueError("Generated trajectory violates envelope")

            self.last_traj = traj
            self.update_plots(traj)

            # status
            if "sdot_peak" in info:
                v_peak = info["sdot_peak"] * dz
                self.status.setText(
                    f"Mode: {info.get('mode','')}\n"
                    f"T_total: {info.get('t_total', 0.0):.6f} s\n"
                    f"sdot_peak: {info['sdot_peak']:.6f}  (vz_peak ≈ {v_peak:.6f} m/s)\n"
                    f"t_cruise: {info.get('t_cruise', 0.0):.6f} s"
                )
            else:
                self.status.setText(
                    f"Mode: {info.get('mode','')}\n"
                    f"T_total: {info.get('t_total', 0.0):.6f} s\n"
                    f"v_peak: {info.get('v_peak', 0.0):.6f} m/s\n"
                    f"t_cruise: {info.get('t_cruise', 0.0):.6f} s"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_plots(self, traj: np.ndarray):
        t = traj[:, 0]
        z = traj[:, 3]
        vz = traj[:, 6]
        az = traj[:, 9]
        jz = traj[:, 12]

        for p in [self.plot_p, self.plot_v, self.plot_a, self.plot_j]:
            p.clear()
            p.showGrid(x=True, y=True)

        self.plot_p.plot(t, z)
        self.plot_p.setLabel("left", "z", units="m")
        self.plot_p.setLabel("bottom", "time", units="s")

        self.plot_v.plot(t, vz)
        self.plot_v.setLabel("left", "vz", units="m/s")
        self.plot_v.setLabel("bottom", "time", units="s")

        self.plot_a.plot(t, az)
        self.plot_a.setLabel("left", "az", units="m/s^2")
        self.plot_a.setLabel("bottom", "time", units="s")

        self.plot_j.plot(t, jz)
        self.plot_j.setLabel("left", "jz", units="m/s^3")
        self.plot_j.setLabel("bottom", "time", units="s")

    def export_csv(self):
        if self.last_traj is None:
            QMessageBox.information(self, "Export", "No trajectory yet")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "traj.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "jx", "jy", "jz"])
            for row in self.last_traj:
                w.writerow([f"{v:.6f}" for v in row])


def main():
    app = QApplication(sys.argv)
    w = ScurvePlayground()
    w.resize(1100, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
