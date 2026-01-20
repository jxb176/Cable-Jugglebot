#!/usr/bin/env python3
"""
demo_juggle_path_xyz_plot.py

Example usage of your JugglePath + LineDVNoCoastScaled primitives:
- Builds a short multi-segment path
- Plots 4 stacked plots (pos/vel/acc/jerk) for x,y,z
- Shows a separate 3D animation of the motion

Assumes you have:
  from juggle_path_xyz import State3D, JugglePath, LineDVNoCoastScaled
in your library module (no __main__ demo in the library file).

Keys in animation window (matches your manual planner feel):
  space : play/pause
  left  : step backward
  right : step forward
  r     : restart
  esc   : close
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Import your library ---
# Adjust this import to match your project structure.
# e.g. from path_primitives.juggle_path_xyz import State3D, JugglePath, LineDVNoCoastScaled
from jugglepath import State3D, JugglePath, LineDVNoCoastScaled


# ----------------------------
# Plotting helper (copied pattern from manual_juggle_planner.py)
# ----------------------------
def place_figure(fig, x: int, y: int, w: int, h: int):
    """Move/resize a matplotlib figure window in screen pixels (Qt/Tk backends)."""
    mgr = fig.canvas.manager
    try:
        mgr.window.setGeometry(x, y, w, h)     # Qt
    except Exception:
        try:
            mgr.window.wm_geometry(f"{w}x{h}+{x}+{y}")  # Tk
        except Exception:
            pass


# ----------------------------
# Plot stacked timeseries
# ----------------------------
def plot_timeseries(traj: np.ndarray, title: str = "JugglePath XYZ kinematics"):
    """
    traj: (N,13) columns [t,x,y,z, vx,vy,vz, ax,ay,az, jx,jy,jz]
    """
    t = traj[:, 0]
    P = traj[:, 1:4]
    V = traj[:, 4:7]
    A = traj[:, 7:10]
    J = traj[:, 10:13]

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(11, 9))
    axs[0].set_title(title)

    # Position
    axs[0].plot(t, P[:, 0], label="x")
    axs[0].plot(t, P[:, 1], label="y")
    axs[0].plot(t, P[:, 2], label="z")
    axs[0].set_ylabel("pos [m]")
    axs[0].legend(loc="upper right", ncol=3)

    # Velocity
    axs[1].plot(t, V[:, 0])
    axs[1].plot(t, V[:, 1])
    axs[1].plot(t, V[:, 2])
    axs[1].set_ylabel("vel [m/s]")

    # Acceleration
    axs[2].plot(t, A[:, 0])
    axs[2].plot(t, A[:, 1])
    axs[2].plot(t, A[:, 2])
    axs[2].set_ylabel("acc [m/s²]")

    # Jerk
    axs[3].plot(t, J[:, 0])
    axs[3].plot(t, J[:, 1])
    axs[3].plot(t, J[:, 2])
    axs[3].set_ylabel("jerk [m/s³]")
    axs[3].set_xlabel("time [s]")

    fig.tight_layout()
    return fig


# ----------------------------
# 3D animation (similar to manual_juggle_planner.py)
# ----------------------------
def animate_xyz(traj: np.ndarray, stride: int = 1, trail: int = 250):
    """
    Simple 3D animation of the XYZ motion.
    - stride: subsampling factor for animation
    - trail: number of samples to show as trail
    """
    T = traj[::stride, 0]
    P = traj[::stride, 1:4]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("JugglePath motion (3D)")

    # Bounds from trajectory with padding
    xmin, ymin, zmin = np.min(P, axis=0)
    xmax, ymax, zmax = np.max(P, axis=0)
    pad = 0.05
    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad
    zmin -= pad; zmax += pad

    # cubic aspect-ish
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    half = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

    # Artists
    marker = ax.plot([], [], [], marker="o", linestyle="None")[0]
    trail_line = ax.plot([], [], [], linewidth=1.5, alpha=0.8)[0]
    full_path = ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=1.0, alpha=0.25)[0]
    _ = full_path  # unused, but keeps a faint full path in view

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    # Playback controls
    state = {"paused": False, "i": 0}

    def set_artists(i: int):
        time_text.set_text(f"t = {float(T[i]):.3f} s")
        p = P[i]

        marker.set_data([p[0]], [p[1]])
        marker.set_3d_properties([p[2]])

        k0 = max(0, i - int(trail))
        tr = P[k0:i + 1]
        trail_line.set_data(tr[:, 0], tr[:, 1])
        trail_line.set_3d_properties(tr[:, 2])

        return [time_text, marker, trail_line]

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "right":
            state["i"] = min(state["i"] + 1, len(T) - 1)
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
            state["i"] = min(state["i"] + 1, len(T) - 1)
        return set_artists(state["i"])

    ani = FuncAnimation(fig, update, interval=20, blit=False)
    return fig, ani


# ----------------------------
# Build a demo path
# ----------------------------
def build_demo_path(sample_hz: float = 500.0):
    start = State3D(p=[0, 0, 0], v=[0, 0, 0], a=[0, 0, 0])

    path = JugglePath(start=start, sample_hz=sample_hz)

    # A small "box-ish" path with different directions
    # v1_along is the commanded along-line terminal speed for each segment.
    accel_ref = 50.0
    jerk_ref = 2000.0

    path.add(LineDVNoCoastScaled(
        p1=[0.0, 0.0, -0.15],
        v1_along=1.0,
        accel_ref=5, jerk_ref=100,
        scale_accel=True, scale_jerk=True
    ))

    path.add(LineDVNoCoastScaled(
        p1=[0.0, 0.0, -0.2],
        v1_along=0.0,
        accel_ref=accel_ref, jerk_ref=jerk_ref,
        scale_accel=True, scale_jerk=True
    ))

    path.add(LineDVNoCoastScaled(
        p1=[0.0, 0.0, 0.0],
        v1_along=6.0,
        accel_ref=accel_ref, jerk_ref=jerk_ref,
        scale_accel=True, scale_jerk=True
    ))

    path.add(LineDVNoCoastScaled(
        p1=[0.0, 0.0, 0.2],
        v1_along=0.0,
        accel_ref=accel_ref, jerk_ref=jerk_ref,
        scale_accel=True, scale_jerk=True
    ))

    path.add(LineDVNoCoastScaled(
        p1=[0.0, 0.0, 0.1],
        v1_along=1.0,
        accel_ref=accel_ref, jerk_ref=jerk_ref,
        scale_accel=True, scale_jerk=True
    ))

    path.add(LineDVNoCoastScaled(
        p1=[0.0, 0.0, 0.0],
        v1_along=0.0,
        accel_ref=accel_ref, jerk_ref=jerk_ref,
        scale_accel=True, scale_jerk=True
    ))

    res = path.build()
    return res


def main():
    res = build_demo_path(sample_hz=500.0)

    print("traj shape:", res.traj.shape)
    print("end state p:", res.end_state.p)
    print("end state v:", res.end_state.v)
    print("end state a:", res.end_state.a)
    print("segment infos:")
    for info in res.segment_infos:
        print(info)

    fig_ts = plot_timeseries(res.traj, title="JugglePath XYZ kinematics (pos/vel/acc/jerk)")
    # Optional: position this window (tune for your monitor layout)
    # place_figure(fig_ts, x=1920, y=900, w=1000, h=900)

    fig_anim, ani = animate_xyz(res.traj, stride=2, trail=250)
    # place_figure(fig_anim, x=1920, y=0, w=1000, h=900)

    plt.show()


if __name__ == "__main__":
    main()
