#!/usr/bin/env python3
"""
demo_juggle_path_xyz_plot.py

Example usage of your JugglePath + LineDVNoCoastScaled primitives:
- Builds a short multi-segment path
- Plots 4 stacked plots (pos/vel/acc/jerk) for x,y,z
- Shows a separate 3D animation of the motion

Keys in animation window (matches your manual planner feel):
  space : play/pause
  left  : step backward
  right : step forward
  r     : restart
  esc   : close
"""

from __future__ import annotations
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from jugglepath import State3D, JugglePath

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
# Path output to pose_cmd.csv for simulation
# ----------------------------
def write_pose_cmd_csv(traj: np.ndarray, out_path: str = "pose_cmd.csv",
                       roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0):
    """
    Export JugglePath traj -> pose_cmd.csv format for MuJoCo sim.

    traj: (N,13) columns [t,x,y,z, vx,vy,vz, ax,ay,az, jx,jy,jz] with SI units.
    out CSV columns: t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg
    """
    if traj.ndim != 2 or traj.shape[1] < 4:
        raise ValueError(f"traj must be (N,>=4). Got {traj.shape}")

    t = traj[:, 0].astype(float)
    xyz_m = traj[:, 1:4].astype(float)
    xyz_mm = 1000.0 * xyz_m

    # Ensure strictly increasing time (your sim loader also dedupes, but let's keep it clean)
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 1e-9
    t = t[keep]
    xyz_mm = xyz_mm[keep]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"])
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.9f}",
                f"{xyz_mm[i,0]:.6f}",
                f"{xyz_mm[i,1]:.6f}",
                f"{xyz_mm[i,2]:.6f}",
                f"{float(roll_deg):.6f}",
                f"{float(pitch_deg):.6f}",
                f"{float(yaw_deg):.6f}",
            ])

def write_pose_cmd_full_csv(traj: np.ndarray, out_path: str = "pose_cmd_full.csv",
                            roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0):
    """
    Export JugglePath traj -> full command CSV for MPC development.

    traj: (N,13) [t,x,y,z,vx,vy,vz,ax,ay,az,jx,jy,jz] (SI units)
    """
    if traj.ndim != 2 or traj.shape[1] < 10:
        raise ValueError(f"traj must be (N,>=10). Got {traj.shape}")

    t = traj[:, 0].astype(float)
    P = traj[:, 1:4].astype(float)
    V = traj[:, 4:7].astype(float)
    A = traj[:, 7:10].astype(float)

    Pmm = 1000.0 * P

    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 1e-9
    t = t[keep]; Pmm = Pmm[keep]; V = V[keep]; A = A[keep]

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "x_mm", "y_mm", "z_mm",
            "vx_mps", "vy_mps", "vz_mps",
            "ax_mps2", "ay_mps2", "az_mps2",
            "roll_deg", "pitch_deg", "yaw_deg",
        ])
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.9f}",
                f"{Pmm[i,0]:.6f}", f"{Pmm[i,1]:.6f}", f"{Pmm[i,2]:.6f}",
                f"{V[i,0]:.6f}",   f"{V[i,1]:.6f}",   f"{V[i,2]:.6f}",
                f"{A[i,0]:.6f}",   f"{A[i,1]:.6f}",   f"{A[i,2]:.6f}",
                f"{float(roll_deg):.6f}",
                f"{float(pitch_deg):.6f}",
                f"{float(yaw_deg):.6f}",
            ])


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

    throw_v = 4.9

    accel_ref = 50.0
    jerk_ref = 2000.0

    # Segment 1: 0 -> -0.2 (down). Limit accel to 0.5g so the ball stays in the hand on initial accel down.
    path.add_segment(
        p=[0.0, 0.0, -0.2],
        v=[0.0, 0.0, 0.0],  # waypoint velocity
        time_law="s_curve",
        accel_ref=5.0,
        jerk_ref=500.0,
    )

    # Segment 2.1: -0.2 -> 0.0 (up). v1_along=+6.0 => +Z => vz=+6.0
    path.add_segment(
        p=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, throw_v],  # waypoint velocity
        time_law="s_curve_monotonic",
        accel_ref=accel_ref,
        jerk_ref=jerk_ref,
    )

    # Segment 2.2: 0.0 -> 0.2 (up). End speed 0.
    path.add_segment(
        p=[0.0, 0.0, 0.2],
        v=[0.0, 0.0, 0.0],  # waypoint velocity
        time_law="s_curve_monotonic",
        accel_ref=accel_ref,
        jerk_ref=jerk_ref,
    )

    # Segment 3: -0.2 -> 0.0 (down), Return to start point and hold
    path.add_segment(
        p=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],  # waypoint velocity
        time_law="s_curve",
        accel_ref=20.0,
        jerk_ref=1000.0,
        v_max=1.0,
    )

    """
    # 3d example
    # Segment 1: 0 -> -0.2 (down). Limit accel to 0.5g so the ball stays in the hand on initial accel down.
    path.add_segment(
        p=[-0.05, 0.0, -0.2],
        v=[0.0, 0.0, 0.0],  # waypoint velocity
        time_law="s_curve",
        accel_ref=5.0,
        jerk_ref=500.0,
    )

    # Segment 2.1: -0.2 -> 0.0 (up). v1_along=+6.0 => +Z => vz=+6.0
    path.add_segment(
        p=[0.0, 0.0, 0.0],
        v=[throw_v/4, 0.0, throw_v],  # waypoint velocity
        time_law="s_curve_monotonic",
        accel_ref=accel_ref,
        jerk_ref=jerk_ref,
    )

    # Segment 2.2: 0.0 -> 0.2 (up). End speed 0.
    path.add_segment(
        p=[0.05, 0.0, 0.2],
        v=[0.0, 0.0, 0.0],  # waypoint velocity
        time_law="s_curve_monotonic",
        accel_ref=accel_ref,
        jerk_ref=jerk_ref,
    )

    # Segment 3: -0.2 -> 0.0 (down), Return to start point and hold
    path.add_segment(
        p=[0.3, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],  # waypoint velocity
        time_law="s_curve",
        accel_ref=20.0,
        jerk_ref=1000.0,
        v_max=1.0,
    )
    
    path.add_wait(duration=2.0)

    """

    res = path.build()
    return res



def main():
    res = build_demo_path(sample_hz=500.0)

    write_pose_cmd_csv(res.traj, out_path="pose_cmd.csv", roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)
    write_pose_cmd_full_csv(res.traj, out_path="pose_cmd_full.csv", roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)

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
