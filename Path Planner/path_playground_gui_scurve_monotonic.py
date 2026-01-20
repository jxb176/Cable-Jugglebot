# path_playground_gui_scurve_monotonic_single_scale_v2.py
# ------------------------------------------------------------
# Z-only jerk-limited S-curve playground (monotonic dv primitive)
#
# This playground ALWAYS generates the same primitive:
#   - Jerk/accel-limited velocity transition from vz0 -> vz1 with a(0)=a(T)=0
#   - No constant-velocity coast phase
#   - Optional scaling of accel_ref and/or jerk_ref to better match the requested dz
#   - Always returns a trajectory segment (planner decides feasibility)
#
# Outputs/plots: position z, velocity vz, acceleration az, jerk jz
# ------------------------------------------------------------

import sys
import math
import csv
from typing import Tuple, List, Optional

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QCheckBox,
    QFileDialog, QGroupBox, QMessageBox
)

import pyqtgraph as pg


# ------------------------------------------------------------
# Exact integration under constant jerk
# State is (p, v, a)
# ------------------------------------------------------------

def _integrate_const_jerk(p: float, v: float, a: float, j: float, dt: float) -> Tuple[float, float, float]:
    """Exact integration for constant jerk over dt."""
    p1 = p + v * dt + 0.5 * a * dt * dt + (1.0 / 6.0) * j * dt**3
    v1 = v + a * dt + 0.5 * j * dt * dt
    a1 = a + j * dt
    return p1, v1, a1


def _build_min_time_dv_segments(v_start: float, v_end: float, amax: float, jmax: float) -> List[Tuple[float, float]]:
    """Min-time jerk-limited, accel-limited velocity transition v_start->v_end.

    Assumes a(0)=0 and a(T)=0.

    Returns a list of (jerk, duration) segments with piecewise-constant jerk.
    """
    dv = v_end - v_start
    if abs(dv) < 1e-15:
        return []

    if amax <= 0 or jmax <= 0:
        return []

    sgn = 1.0 if dv > 0 else -1.0
    dv = abs(dv)

    # Triangular accel: a_peak = sqrt(dv * jmax)
    a_peak = math.sqrt(dv * jmax)

    if a_peak <= amax + 1e-15:
        Tj = a_peak / jmax
        return [(sgn * jmax, Tj), (-sgn * jmax, Tj)]

    # Trapezoidal accel (hits amax)
    Tj = amax / jmax
    Ta = dv / amax - Tj
    if Ta < 0:
        Ta = 0.0

    segs: List[Tuple[float, float]] = [(sgn * jmax, Tj)]
    if Ta > 0:
        segs.append((0.0, Ta))
    segs.append((-sgn * jmax, Tj))
    return segs


def _simulate_segments(
    segments: List[Tuple[float, float]],
    p0: float,
    v0: float,
    a0: float,
    dt: float,
) -> np.ndarray:
    """Sample rows [t, p, v, a, j] while integrating piecewise-constant jerk."""
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
    """Return (distance, time, segments) for the min-time dv transition."""
    segs = _build_min_time_dv_segments(v_start, v_end, amax, jmax)

    p = 0.0
    v = v_start
    a = 0.0
    t = 0.0
    for (j, dur) in segs:
        p, v, a = _integrate_const_jerk(p, v, a, j, dur)
        t += dur

    return p, t, segs


# ------------------------------------------------------------
# Primitive: monotonic dv transition, no coasting, with optional scaling
# ------------------------------------------------------------

def generate_z_profile_dv_no_coast_scaled(
    z0: float,
    z1: float,
    vz0: float,
    vz1: float,
    accel_ref: float,
    jerk_ref: float,
    sample_hz: float,
    scale_accel: bool,
    scale_jerk: bool,
    k_min: float = 1e-4,
    k_max: float = 1e4,
    grid_points: int = 81,
) -> Tuple[np.ndarray, dict]:
    """Generate a Z trajectory using only a dv transition (no cruise).

    We optionally apply a single scalar k to accel_ref and/or jerk_ref:
      a_used = accel_ref * k if scale_accel else accel_ref
      j_used = jerk_ref  * k if scale_jerk  else jerk_ref

    k is chosen to minimize | |d_base(k)| - |dz| |, where d_base is the
    dv-transition displacement from integrating the min-time dv segment.

    Important: This function ALWAYS returns a trajectory. It does not throw
    infeasibility errors based on dz, endpoint velocities, etc. Any mismatch is
    reported in info so the higher-level planner can decide what to do.

    The returned trajectory always ends exactly at z1 (we apply a constant
    position offset); velocities/accelerations/jerks are unaffected by that
    correction.
    """
    if sample_hz <= 0:
        raise ValueError("sample_hz must be > 0")

    z0 = float(z0)
    z1 = float(z1)
    vz0 = float(vz0)
    vz1 = float(vz1)
    accel_ref = float(accel_ref)
    jerk_ref = float(jerk_ref)

    if accel_ref <= 0 or jerk_ref <= 0:
        raise ValueError("accel_ref and jerk_ref must be > 0")

    dt = 1.0 / float(sample_hz)
    dz = z1 - z0

    # If neither scaling is enabled, k is fixed at 1.
    if not scale_accel and not scale_jerk:
        k_used = 1.0
    else:
        # Clamp and ensure reasonable bounds
        k_min = max(1e-12, float(k_min))
        k_max = max(k_min * 1.0001, float(k_max))
        grid_points = int(max(11, grid_points))

        # Sample k log-spaced to find best match
        ks = np.logspace(math.log10(k_min), math.log10(k_max), grid_points)
        dz_abs = abs(dz)

        best_k = float(ks[0])
        best_err = float("inf")
        best_d = 0.0

        for k in ks:
            a_used = accel_ref * k if scale_accel else accel_ref
            j_used = jerk_ref * k if scale_jerk else jerk_ref
            d_base, _, _ = _phase_distance_time(vz0, vz1, a_used, j_used)

            err = abs(abs(d_base) - dz_abs)
            if err < best_err:
                best_err = err
                best_k = float(k)
                best_d = float(d_base)

        # Local refine around best_k with a few bisection iterations if possible.
        # We try to bracket a sign change for f(k)=|d(k)|-|dz|.
        def f(k: float) -> float:
            a_used = accel_ref * k if scale_accel else accel_ref
            j_used = jerk_ref * k if scale_jerk else jerk_ref
            d_base, _, _ = _phase_distance_time(vz0, vz1, a_used, j_used)
            return abs(d_base) - dz_abs

        k_used = best_k
        f_mid = f(k_used)

        # Choose neighbors from the sampled grid
        idx = int(np.argmin(np.abs(ks - best_k)))
        lo = float(ks[max(0, idx - 1)])
        hi = float(ks[min(len(ks) - 1, idx + 1)])
        f_lo = f(lo)
        f_hi = f(hi)

        # If we have a bracket, bisection to reduce error.
        if f_lo == 0.0:
            k_used = lo
        elif f_hi == 0.0:
            k_used = hi
        elif f_lo * f_hi < 0.0:
            a = lo
            b = hi
            fa = f_lo
            fb = f_hi
            for _ in range(40):
                m = 0.5 * (a + b)
                fm = f(m)
                # Keep smaller interval
                if fa * fm <= 0.0:
                    b = m
                    fb = fm
                else:
                    a = m
                    fa = fm
            k_used = 0.5 * (a + b)

    # Build final profile at k_used
    a_used = accel_ref * k_used if scale_accel else accel_ref
    j_used = jerk_ref * k_used if scale_jerk else jerk_ref

    d_base, t_base, segs = _phase_distance_time(vz0, vz1, a_used, j_used)
    samples = _simulate_segments(segs, p0=z0, v0=vz0, a0=0.0, dt=dt)

    if samples.shape[0] == 0:
        # Degenerate: return a single sample at t=0
        traj = np.zeros((1, 13), dtype=float)
        traj[0, 0] = 0.0
        traj[0, 3] = z0
        traj[0, 6] = vz0
        info = {
            "mode": "dv_no_coast_scaled",
            "k_used": float(k_used),
            "accel_ref_used": float(a_used),
            "jerk_ref_used": float(j_used),
            "d_base": float(d_base),
            "dz": float(dz),
            "t_total": 0.0,
            "degenerate": True,
        }
        return traj, info

    # Force end position to z1 by constant offset (does not change v/a/j)
    z_end = float(samples[-1, 1])
    samples[:, 1] += (z1 - z_end)

    traj = np.zeros((samples.shape[0], 13), dtype=float)
    traj[:, 0] = samples[:, 0]
    traj[:, 3] = samples[:, 1]
    traj[:, 6] = samples[:, 2]
    traj[:, 9] = samples[:, 3]
    traj[:, 12] = samples[:, 4]

    info = {
        "mode": "dv_no_coast_scaled",
        "k_used": float(k_used),
        "scale_accel": bool(scale_accel),
        "scale_jerk": bool(scale_jerk),
        "accel_ref_used": float(a_used),
        "jerk_ref_used": float(j_used),
        "a_peak": float(np.max(np.abs(traj[:, 9]))),
        "j_peak": float(np.max(np.abs(traj[:, 12]))),
        "t_total": float(traj[-1, 0]),
        "t_base": float(t_base),
        "d_base": float(d_base),
        "dz": float(dz),
        "d_error_abs": float(abs(abs(d_base) - abs(dz))),
    }
    return traj, info


# ------------------------------------------------------------
# GUI
# ------------------------------------------------------------

class ScurvePlayground(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z S-curve Playground (dv primitive, no coast)")

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
        self.z0 = QDoubleSpinBox(); self.z0.setRange(-10, 10); self.z0.setDecimals(6); self.z0.setValue(-0.25)
        g.addWidget(self.z0, row, 1); row += 1

        g.addWidget(QLabel("z1 [m]"), row, 0)
        self.z1 = QDoubleSpinBox(); self.z1.setRange(-10, 10); self.z1.setDecimals(6); self.z1.setValue(0.0)
        g.addWidget(self.z1, row, 1); row += 1

        g.addWidget(QLabel("vz0 [m/s]"), row, 0)
        self.vz0 = QDoubleSpinBox(); self.vz0.setRange(-1e4, 1e4); self.vz0.setDecimals(6); self.vz0.setValue(0.0)
        g.addWidget(self.vz0, row, 1); row += 1

        g.addWidget(QLabel("vz1 [m/s]"), row, 0)
        self.vz1 = QDoubleSpinBox(); self.vz1.setRange(-1e4, 1e4); self.vz1.setDecimals(6); self.vz1.setValue(5.0)
        g.addWidget(self.vz1, row, 1); row += 1

        # ---- Limits ----
        lg = QGroupBox("Reference dynamics")
        left.addWidget(lg)
        lgd = QGridLayout(lg)

        lgd.addWidget(QLabel("accel_ref [m/s²]"), 0, 0)
        self.accel_ref = QDoubleSpinBox(); self.accel_ref.setRange(1e-6, 1e9); self.accel_ref.setDecimals(6); self.accel_ref.setValue(50.0)
        lgd.addWidget(self.accel_ref, 0, 1)

        lgd.addWidget(QLabel("jerk_ref [m/s³]"), 1, 0)
        self.jerk_ref = QDoubleSpinBox(); self.jerk_ref.setRange(1e-6, 1e12); self.jerk_ref.setDecimals(6); self.jerk_ref.setValue(2000.0)
        lgd.addWidget(self.jerk_ref, 1, 1)

        lgd.addWidget(QLabel("Sample Hz"), 2, 0)
        self.sample_hz = QDoubleSpinBox(); self.sample_hz.setRange(10, 5000); self.sample_hz.setDecimals(1); self.sample_hz.setValue(1000.0)
        lgd.addWidget(self.sample_hz, 2, 1)

        # ---- Scaling options ----
        sg = QGroupBox("Optional scaling")
        left.addWidget(sg)
        sgd = QGridLayout(sg)

        self.scale_accel = QCheckBox("Scale accel_ref")
        self.scale_accel.setChecked(True)
        sgd.addWidget(self.scale_accel, 0, 0, 1, 2)

        self.scale_jerk = QCheckBox("Scale jerk_ref")
        self.scale_jerk.setChecked(True)
        sgd.addWidget(self.scale_jerk, 1, 0, 1, 2)

        sgd.addWidget(QLabel("k_min"), 2, 0)
        self.k_min = QDoubleSpinBox(); self.k_min.setRange(1e-12, 1e6); self.k_min.setDecimals(8); self.k_min.setValue(1e-4)
        sgd.addWidget(self.k_min, 2, 1)

        sgd.addWidget(QLabel("k_max"), 3, 0)
        self.k_max = QDoubleSpinBox(); self.k_max.setRange(1e-12, 1e12); self.k_max.setDecimals(8); self.k_max.setValue(1e4)
        sgd.addWidget(self.k_max, 3, 1)

        sgd.addWidget(QLabel("Grid points"), 4, 0)
        self.grid_points = QDoubleSpinBox(); self.grid_points.setRange(11, 401); self.grid_points.setDecimals(0); self.grid_points.setValue(81)
        sgd.addWidget(self.grid_points, 4, 1)

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

        self.last_traj: Optional[np.ndarray] = None

        self.btn_run.clicked.connect(self.run)
        self.btn_export.clicked.connect(self.export_csv)

    def run(self):
        try:
            z0 = float(self.z0.value())
            z1 = float(self.z1.value())
            vz0 = float(self.vz0.value())
            vz1 = float(self.vz1.value())
            accel_ref = float(self.accel_ref.value())
            jerk_ref = float(self.jerk_ref.value())
            sample_hz = float(self.sample_hz.value())

            traj, info = generate_z_profile_dv_no_coast_scaled(
                z0=z0,
                z1=z1,
                vz0=vz0,
                vz1=vz1,
                accel_ref=accel_ref,
                jerk_ref=jerk_ref,
                sample_hz=sample_hz,
                scale_accel=self.scale_accel.isChecked(),
                scale_jerk=self.scale_jerk.isChecked(),
                k_min=float(self.k_min.value()),
                k_max=float(self.k_max.value()),
                grid_points=int(self.grid_points.value()),
            )

            self.last_traj = traj
            self.update_plots(traj)

            self.status.setText(
                f"Mode: {info.get('mode','')}\n"
                f"T_total: {info.get('t_total',0.0):.6f} s\n"
                f"k_used: {info.get('k_used',1.0):.6g}  (scale_accel={info.get('scale_accel',False)}, scale_jerk={info.get('scale_jerk',False)})\n"
                f"accel_ref_used: {info.get('accel_ref_used',0.0):.6g}  jerk_ref_used: {info.get('jerk_ref_used',0.0):.6g}\n"
                f"a_peak: {info.get('a_peak',0.0):.6g}  j_peak: {info.get('j_peak',0.0):.6g}\n"
                f"dz: {info.get('dz',0.0):.6g}  d_base: {info.get('d_base',0.0):.6g}  |error|: {info.get('d_error_abs',0.0):.6g}"
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
                w.writerow([f"{v:.10g}" for v in row])


def main():
    app = QApplication(sys.argv)
    w = ScurvePlayground()
    w.resize(1100, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
