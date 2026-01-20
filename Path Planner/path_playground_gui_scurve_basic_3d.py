# path_playground_gui_scurve_basic_3d.py
# ------------------------------------------------------------
# 3D S-curve path primitive (single scalar s(t) mapped onto 3D line segment)
#
# Path:
#   p(s) = p0 + s*(p1-p0),  s in [0,1]
# Time-law (S-curve):
#   s(t), sdot, sddot, sjerk jerk-limited
#
# Mapping:
#   v = sdot * d
#   a = sddot * d
#   j = sjerk * d
#
# Endpoint velocity vectors v0/v1 are accepted but only the COMPONENT ALONG the path direction
# can be matched. We project v0/v1 onto d and warn if a perpendicular component is requested.
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


# -------------------- Helpers: exact integration under constant jerk --------------------

def _integrate_const_jerk(p: float, v: float, a: float, j: float, dt: float):
    p1 = p + v*dt + 0.5*a*dt*dt + (1.0/6.0)*j*dt**3
    v1 = v + a*dt + 0.5*j*dt*dt
    a1 = a + j*dt
    return p1, v1, a1


def _build_min_time_dv_segments(v_start: float, v_end: float, amax: float, jmax: float) -> List[Tuple[float, float]]:
    dv = v_end - v_start
    if abs(dv) < 1e-15:
        return []

    sgn = 1.0 if dv > 0 else -1.0
    dv = abs(dv)

    a_peak = math.sqrt(dv * jmax)
    if a_peak <= amax + 1e-15:
        Tj = a_peak / jmax
        return [(sgn*jmax, Tj), (-sgn*jmax, Tj)]

    Tj = amax / jmax
    Ta = dv/amax - Tj
    if Ta < 0:
        Ta = 0.0

    segs = [(sgn*jmax, Tj)]
    if Ta > 0:
        segs.append((0.0, Ta))
    segs.append((-sgn*jmax, Tj))
    return segs


def _simulate_segments(segments: List[Tuple[float, float]], p0: float, v0: float, a0: float, dt: float) -> np.ndarray:
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


def _phase_distance_time(v_start: float, v_end: float, amax: float, jmax: float) -> Tuple[float, float]:
    segs = _build_min_time_dv_segments(v_start, v_end, amax, jmax)
    p = 0.0
    v = v_start
    a = 0.0
    t = 0.0
    for (j, dur) in segs:
        p, v, a = _integrate_const_jerk(p, v, a, j, dur)
        t += dur
    return p, t


# -------------------- Normalized time-law planner: s(t) in [0,1] --------------------

def generate_scurve_normalized(
    sdot0: float,
    sdot1: float,
    sdot_max: float,
    amax: float,
    jmax: float,
    sample_hz: float
) -> Tuple[np.ndarray, dict]:
    """
    Classic structure:
      sdot0 -> sdot_peak (min-time)
      optional cruise at sdot_peak
      sdot_peak -> sdot1 (min-time)

    Returns samples [t, s, sdot, sddot, sjerk].
    """
    if sample_hz <= 0:
        raise ValueError("sample_hz must be > 0")
    if amax <= 0 or jmax <= 0 or sdot_max <= 0:
        raise ValueError("limits must be > 0")

    if abs(sdot0) - sdot_max > 1e-9 or abs(sdot1) - sdot_max > 1e-9:
        raise ValueError("endpoint |sdot| exceeds sdot_max")

    # choose nonnegative peak for simplicity
    v_lo = max(0.0, sdot0, sdot1)
    v_hi = float(sdot_max)

    def d_min_for_peak(vp: float) -> float:
        d1, _ = _phase_distance_time(sdot0, vp, amax, jmax)
        d2, _ = _phase_distance_time(vp, sdot1, amax, jmax)
        return d1 + d2

    d_at_hi = d_min_for_peak(v_hi)

    if d_at_hi <= 1.0 + 1e-12:
        v_peak = v_hi
        d_min = d_at_hi
        t_cruise = (1.0 - d_min) / v_peak if v_peak > 1e-15 else 0.0
        mode = "cruise_at_vmax" if t_cruise > 0 else "no_cruise_at_vmax"
    else:
        d_at_lo = d_min_for_peak(v_lo)
        if d_at_lo > 1.0 + 1e-12:
            raise ValueError("infeasible (need more distance or higher limits)")

        lo, hi = v_lo, v_hi
        for _ in range(90):
            mid = 0.5*(lo + hi)
            if d_min_for_peak(mid) > 1.0:
                hi = mid
            else:
                lo = mid
            if abs(d_min_for_peak(mid) - 1.0) < 1e-12:
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

    dt = 1.0 / sample_hz
    samp = _simulate_segments(segments, p0=0.0, v0=sdot0, a0=0.0, dt=dt)
    if samp.shape[0] == 0:
        raise ValueError("degenerate profile")

    s_end = samp[-1, 1]
    if abs(s_end) < 1e-15:
        raise ValueError("near-zero distance")

    # normalize so s_end == 1
    scale = 1.0 / s_end
    samp[:, 1] *= scale
    samp[:, 2] *= scale
    samp[:, 3] *= scale
    samp[:, 4] *= scale

    info = {"mode": mode, "sdot_peak": float(v_peak), "t_total": float(samp[-1, 0]), "t_cruise": float(t_cruise)}
    return samp, info


def map_scurve_to_3d(sc: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """
    Produce traj columns:
      t, x,y,z, vx,vy,vz, ax,ay,az, jx,jy,jz
    """
    d = (p1 - p0).astype(float)
    traj = np.zeros((sc.shape[0], 13), dtype=float)
    traj[:, 0] = sc[:, 0]

    # position
    P = p0[None, :] + sc[:, 1:2] * d[None, :]
    traj[:, 1] = P[:, 0]
    traj[:, 2] = P[:, 1]
    traj[:, 3] = P[:, 2]

    # velocity
    V = sc[:, 2:3] * d[None, :]
    traj[:, 4] = V[:, 0]
    traj[:, 5] = V[:, 1]
    traj[:, 6] = V[:, 2]

    # acceleration
    A = sc[:, 3:4] * d[None, :]
    traj[:, 7] = A[:, 0]
    traj[:, 8] = A[:, 1]
    traj[:, 9] = A[:, 2]

    # jerk
    J = sc[:, 4:5] * d[None, :]
    traj[:, 10] = J[:, 0]
    traj[:, 11] = J[:, 1]
    traj[:, 12] = J[:, 2]

    return traj


# -------------------- GUI --------------------

class Scurve3DPlayground(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D S-curve Playground (line segment)")

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 0)
        root.addLayout(right, 1)

        # ---- Start/End ----
        cg = QGroupBox("Start/End")
        left.addWidget(cg)
        g = QGridLayout(cg)

        def spin(val=0.0, dec=4, lo=-10, hi=10):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setValue(val)
            return s

        row = 0
        g.addWidget(QLabel("x0 [m]"), row, 0); self.x0 = spin(0.0); g.addWidget(self.x0, row, 1)
        g.addWidget(QLabel("x1 [m]"), row, 2); self.x1 = spin(0.3); g.addWidget(self.x1, row, 3); row += 1

        g.addWidget(QLabel("y0 [m]"), row, 0); self.y0 = spin(0.0); g.addWidget(self.y0, row, 1)
        g.addWidget(QLabel("y1 [m]"), row, 2); self.y1 = spin(0.0); g.addWidget(self.y1, row, 3); row += 1

        g.addWidget(QLabel("z0 [m]"), row, 0); self.z0 = spin(0.0); g.addWidget(self.z0, row, 1)
        g.addWidget(QLabel("z1 [m]"), row, 2); self.z1 = spin(0.5); g.addWidget(self.z1, row, 3); row += 1

        # ---- Endpoint velocities (vector) ----
        vg = QGroupBox("Endpoint velocity vectors (will be projected onto path direction)")
        left.addWidget(vg)
        vg2 = QGridLayout(vg)

        row = 0
        vg2.addWidget(QLabel("vx0 [m/s]"), row, 0); self.vx0 = spin(0.0, dec=4, lo=-100, hi=100); vg2.addWidget(self.vx0, row, 1)
        vg2.addWidget(QLabel("vx1 [m/s]"), row, 2); self.vx1 = spin(0.0, dec=4, lo=-100, hi=100); vg2.addWidget(self.vx1, row, 3); row += 1

        vg2.addWidget(QLabel("vy0 [m/s]"), row, 0); self.vy0 = spin(0.0, dec=4, lo=-100, hi=100); vg2.addWidget(self.vy0, row, 1)
        vg2.addWidget(QLabel("vy1 [m/s]"), row, 2); self.vy1 = spin(0.0, dec=4, lo=-100, hi=100); vg2.addWidget(self.vy1, row, 3); row += 1

        vg2.addWidget(QLabel("vz0 [m/s]"), row, 0); self.vz0 = spin(0.0, dec=4, lo=-100, hi=100); vg2.addWidget(self.vz0, row, 1)
        vg2.addWidget(QLabel("vz1 [m/s]"), row, 2); self.vz1 = spin(0.0, dec=4, lo=-100, hi=100); vg2.addWidget(self.vz1, row, 3); row += 1

        # ---- Limits ----
        lg = QGroupBox("Limits (applied per-axis; converted to scalar s-limits)")
        left.addWidget(lg)
        lgd = QGridLayout(lg)

        lgd.addWidget(QLabel("v_max [m/s]"), 0, 0)
        self.vmax = QDoubleSpinBox(); self.vmax.setRange(0.001, 1000); self.vmax.setDecimals(4); self.vmax.setValue(3.0)
        lgd.addWidget(self.vmax, 0, 1)

        lgd.addWidget(QLabel("a_max [m/s²]"), 1, 0)
        self.amax = QDoubleSpinBox(); self.amax.setRange(0.01, 1e6); self.amax.setDecimals(3); self.amax.setValue(50.0)
        lgd.addWidget(self.amax, 1, 1)

        lgd.addWidget(QLabel("j_max [m/s³]"), 2, 0)
        self.jmax = QDoubleSpinBox(); self.jmax.setRange(0.01, 1e6); self.jmax.setDecimals(3); self.jmax.setValue(200.0)
        lgd.addWidget(self.jmax, 2, 1)

        lgd.addWidget(QLabel("Sample Hz"), 3, 0)
        self.sample_hz = QDoubleSpinBox(); self.sample_hz.setRange(10, 2000); self.sample_hz.setValue(200)
        lgd.addWidget(self.sample_hz, 3, 1)

        # Buttons
        self.btn_run = QPushButton("Generate")
        self.btn_export = QPushButton("Export CSV")
        left.addWidget(self.btn_run)
        left.addWidget(self.btn_export)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        # ---- Plots (show Z + speed magnitude, etc.) ----
        pg.setConfigOptions(antialias=True)
        self.plot_p = pg.PlotWidget(title="Position (x,y,z)")
        self.plot_v = pg.PlotWidget(title="Velocity (vx,vy,vz)")
        self.plot_a = pg.PlotWidget(title="Acceleration (ax,ay,az)")
        self.plot_j = pg.PlotWidget(title="Jerk (jx,jy,jz)")

        right.addWidget(self.plot_p, 1)
        right.addWidget(self.plot_v, 1)
        right.addWidget(self.plot_a, 1)
        right.addWidget(self.plot_j, 1)

        self.last_traj = None
        self.btn_run.clicked.connect(self.run)
        self.btn_export.clicked.connect(self.export_csv)

    def run(self):
        try:
            p0 = np.array([self.x0.value(), self.y0.value(), self.z0.value()], dtype=float)
            p1 = np.array([self.x1.value(), self.y1.value(), self.z1.value()], dtype=float)
            d = p1 - p0
            L = float(np.linalg.norm(d))
            if L < 1e-12:
                raise ValueError("p0 and p1 are identical. For dz==0-style velocity-change-at-same-point, we need a separate 3D excursion primitive next.")

            v0 = np.array([self.vx0.value(), self.vy0.value(), self.vz0.value()], dtype=float)
            v1 = np.array([self.vx1.value(), self.vy1.value(), self.vz1.value()], dtype=float)

            vmax = float(self.vmax.value())
            amax = float(self.amax.value())
            jmax = float(self.jmax.value())
            sample_hz = float(self.sample_hz.value())

            u = d / L  # unit direction along segment

            # Project endpoint velocities onto segment direction
            v0_par = float(np.dot(v0, u))
            v1_par = float(np.dot(v1, u))
            v0_perp = float(np.linalg.norm(v0 - v0_par*u))
            v1_perp = float(np.linalg.norm(v1 - v1_par*u))

            warn = ""
            tol = 1e-3
            if v0_perp > tol or v1_perp > tol:
                warn = f"\nWARNING: This primitive is a straight line; perpendicular endpoint velocity components cannot be matched.\n" \
                       f"  |v0_perp|={v0_perp:.6f} m/s, |v1_perp|={v1_perp:.6f} m/s\n" \
                       f"  Using projected along-path components: v0_par={v0_par:.6f}, v1_par={v1_par:.6f}\n"

            # Convert to s-space:
            # Along path, speed = sdot * L  => sdot = v_par / L
            sdot0 = v0_par / L
            sdot1 = v1_par / L

            # Limits: per-axis magnitude limits translate into a conservative scalar limit.
            # We use the fact that component i of velocity is v_i = sdot * d_i.
            # So |sdot| <= vmax/|d_i| for each axis with |d_i|>0 -> take min.
            eps = 1e-12
            sdot_max = float("inf")
            a_max_s = float("inf")
            j_max_s = float("inf")
            for di in d:
                adi = abs(di)
                if adi > eps:
                    sdot_max = min(sdot_max, vmax / adi)
                    a_max_s = min(a_max_s, amax / adi)
                    j_max_s = min(j_max_s, jmax / adi)

            if not np.isfinite(sdot_max) or not np.isfinite(a_max_s) or not np.isfinite(j_max_s):
                raise ValueError("Bad limits computed (check p0/p1).")

            sc, info = generate_scurve_normalized(
                sdot0=sdot0, sdot1=sdot1,
                sdot_max=sdot_max,
                amax=a_max_s, jmax=j_max_s,
                sample_hz=sample_hz
            )

            traj = map_scurve_to_3d(sc, p0, p1)
            self.last_traj = traj
            self.update_plots(traj)

            v_peak = info["sdot_peak"] * L
            self.status.setText(
                f"Mode: {info.get('mode','')}\n"
                f"T_total: {info.get('t_total', 0.0):.6f} s\n"
                f"v_peak_along_path ≈ {v_peak:.6f} m/s\n"
                f"t_cruise: {info.get('t_cruise', 0.0):.6f} s"
                f"{warn}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_plots(self, traj: np.ndarray):
        t = traj[:, 0]
        x, y, z = traj[:, 1], traj[:, 2], traj[:, 3]
        vx, vy, vz = traj[:, 4], traj[:, 5], traj[:, 6]
        ax, ay, az = traj[:, 7], traj[:, 8], traj[:, 9]
        jx, jy, jz = traj[:,10], traj[:,11], traj[:,12]

        for p in [self.plot_p, self.plot_v, self.plot_a, self.plot_j]:
            p.clear()
            p.showGrid(x=True, y=True)

        self.plot_p.plot(t, x, name="x")
        self.plot_p.plot(t, y, name="y")
        self.plot_p.plot(t, z, name="z")
        self.plot_p.setLabel("bottom", "time", units="s")

        self.plot_v.plot(t, vx, name="vx")
        self.plot_v.plot(t, vy, name="vy")
        self.plot_v.plot(t, vz, name="vz")
        self.plot_v.setLabel("bottom", "time", units="s")

        self.plot_a.plot(t, ax, name="ax")
        self.plot_a.plot(t, ay, name="ay")
        self.plot_a.plot(t, az, name="az")
        self.plot_a.setLabel("bottom", "time", units="s")

        self.plot_j.plot(t, jx, name="jx")
        self.plot_j.plot(t, jy, name="jy")
        self.plot_j.plot(t, jz, name="jz")
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
            w.writerow(["t","x","y","z","vx","vy","vz","ax","ay","az","jx","jy","jz"])
            for row in self.last_traj:
                w.writerow([f"{v:.6f}" for v in row])


def main():
    app = QApplication(sys.argv)
    w = Scurve3DPlayground()
    w.resize(1200, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
