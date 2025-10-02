import numpy as np
import mujoco
import mujoco.viewer

import matplotlib
matplotlib.use("TkAgg")   # must be before pyplot import
import matplotlib.pyplot as plt

plt.ion()  # interactive mode


# Load model
model = mujoco.MjModel.from_xml_path("cable_robot.xml")
data = mujoco.MjData(model)

# Get platform body ID
platform_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "platform")

# Anchors and platform sites (same as before)
anchors = np.array([
    [ 0.5,  0.5, 1.0],
    [ 0.5, -0.5, 1.0],
    [-0.5,  0.5, 1.0],
    [-0.5, -0.5, 1.0],
    [ 0.5,  0.5, 0.0],
    [-0.5, -0.5, 0.0]
])
platform_sites = np.array([
    [ 0.05,  0.05, 0.0],
    [ 0.05, -0.05, 0.0],
    [-0.05,  0.05, 0.0],
    [-0.05, -0.05, 0.0],
    [ 0.05,  0.05, 0.0],
    [-0.05, -0.05, 0.0],
])

def compute_cable_lengths(platform_pos):
    lengths = []
    for anchor, offset in zip(anchors, platform_sites):
        site_world = platform_pos + offset
        L = np.linalg.norm(site_world - anchor)
        lengths.append(L)
    return np.array(lengths)

def complex_trajectory(t):
    """3D path: Lissajous curve in xy, sinusoid in z."""
    x = 0.15 * np.sin(2*np.pi*0.2*t)
    y = 0.15 * np.sin(2*np.pi*0.3*t + np.pi/2)
    z = 0.5 + 0.1 * np.sin(2*np.pi*0.25*t)
    return np.array([x, y, z])

# Logging
log_time, log_cmd, log_actual = [], [], []

# Run sim ~20s
sim_time = 20.0
with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = data.time
    while viewer.is_running() and data.time - t0 < sim_time:
        t = data.time - t0

        # Commanded trajectory
        platform_pos_des = complex_trajectory(t)

        # Cable lengths from IK
        desired_lengths = compute_cable_lengths(platform_pos_des)
        data.ctrl[:] = desired_lengths

        # Step sim
        mujoco.mj_step(model, data)
        viewer.sync()

        # Log
        actual_pos = data.xpos[platform_id].copy()
        log_time.append(t)
        log_cmd.append(platform_pos_des)
        log_actual.append(actual_pos)

# Convert to arrays
log_time = np.array(log_time)
log_cmd = np.array(log_cmd)
log_actual = np.array(log_actual)

# ==========================
# Plot time histories
# ==========================
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
labels = ["x [m]", "y [m]", "z [m]"]

for i in range(3):
    axs[i].plot(log_time, log_cmd[:, i], "r--", label="Commanded")
    axs[i].plot(log_time, log_actual[:, i], "b-", label="Actual")
    axs[i].set_ylabel(labels[i])
    axs[i].grid(True)
    if i == 0:
        axs[i].legend()

axs[-1].set_xlabel("Time [s]")
plt.suptitle("Platform Commanded vs Actual (time domain)")
plt.tight_layout()
plt.show()  # keep the figure open

# ==========================
# Plot 3D trajectory
# ==========================
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot(log_cmd[:,0], log_cmd[:,1], log_cmd[:,2], "r--", label="Commanded")
ax.plot(log_actual[:,0], log_actual[:,1], log_actual[:,2], "b-", label="Actual")

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("3D Trajectory: Commanded vs Actual")
ax.legend()
plt.show(block=True)  # keep the figure open
