# Cable Robot Simulation (Stiffness-Only Model, recalc L_nat each step)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.append(r"c:\Sandbox\JugglePath")
from JugglePath import JugglePath

# ------------------------
# Anchors and platform geometry
# ------------------------
anchors = np.array([
    [ 0.5243,  0.0000,  0.4389],
    [ 0.5243,  0.0000, -0.4389],
    [-0.2621,  0.4540,  0.4389],
    [-0.2621,  0.4540, -0.4389],
    [-0.2621, -0.4540,  0.4389],
    [-0.2621, -0.4540, -0.4389]
])

platform_nodes = np.array([
    [ 0.0579,  0.0000,  0.0000],
    [ 0.0494,  0.0000, -0.0112],
    [-0.0290,  0.0502,  0.0000],
    [-0.0247,  0.0428, -0.0112],
    [-0.0290, -0.0502,  0.0000],
    [-0.0247, -0.0428, -0.0112]
])
n = anchors.shape[0]

# ------------------------
# Trajectory (JugglePath)
# ------------------------
t_wp = np.array([0.0, 0.15, 0.3])   # start, mid, end times
x_wp = np.array([
    [0.0, 0.0, 0.0],    # x path
    [0.0, 0.0, 0.0],    # y path
    [-0.25, 0.0, 0.25]  # z path
])
v_wp = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 4.4, 0.0]
])
a_wp = np.zeros((3,3))
j_wp = np.zeros((3,3))

traj = JugglePath("poly", t_wp, x_wp, v_wp, a_wp, j_wp)

# ------------------------
# Sampling
# ------------------------
dt = 0.005
ts = np.arange(0.0, t_wp[-1], dt)   # [0, 0.3)
N = len(ts)

platform_pos = []
cable_L = []

for ti in ts:
    p = traj.get_x(ti)
    if p is None:
        continue
    platform_pos.append(p)

    # Cable geometry
    nodes_world = p + platform_nodes
    vec = anchors - nodes_world
    L = np.linalg.norm(vec, axis=1)
    cable_L.append(L)

platform_pos = np.array(platform_pos)
cable_L      = np.array(cable_L)

# ------------------------
# Cable stiffness model (recalc L_nat each step)
# ------------------------
EA   = 1.39e5                 # axial stiffness [N]
pretension_offset = 0.001     # shorten rest length by 1 mm

cable_T = np.zeros((N, n))
cable_L_nat = np.zeros((N, n))

for k in range(N):
    Lk = cable_L[k]

    # Recalculate natural length at this timestep
    L_nat_k = Lk - pretension_offset
    cable_L_nat[k, :] = L_nat_k

    # Extension relative to this natural length
    dL = Lk - L_nat_k
    strain = np.maximum(dL / L_nat_k, 0.0)
    T_stiff = EA * strain
    cable_T[k, :] = T_stiff

# ------------------------
# Sum of Z-force from stiffness
# ------------------------
Fz_stiff = np.zeros(N)
for k in range(N):
    nodes_world = platform_pos[k] + platform_nodes
    vec = anchors - nodes_world
    Lk = np.linalg.norm(vec, axis=1)
    Uk = (vec.T / Lk).T
    Fz_stiff[k] = np.sum(cable_T[k, :] * Uk[:, 2])

plt.figure(figsize=(8,4))
plt.plot(ts, Fz_stiff, label="Cable Z-Force (Stiffness only)")
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("Platform Z-Force (Stiffness Model Only)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ------------------------
# Cable tensions plot
# ------------------------
plt.figure(figsize=(9,5))
for i in range(n):
    plt.plot(ts, cable_T[:, i], label=f"T{i+1}")
plt.xlabel("Time [s]")
plt.ylabel("Tension [N]")
plt.title("Cable Tensions (Stiffness Only)")
plt.legend(ncol=3, fontsize=9)
plt.grid(True, alpha=0.3)
plt.show()

# ------------------------
# Visualization at final pose
# ------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pos = platform_pos[-1]
nodes_world = pos + platform_nodes

ax.scatter(anchors[:,0], anchors[:,1], anchors[:,2], c="k", marker="o", label="anchors")
ax.scatter(nodes_world[:,0], nodes_world[:,1], nodes_world[:,2], c="orange", marker="^", label="platform nodes")
for i in range(n):
    ax.plot([anchors[i,0], nodes_world[i,0]],
            [anchors[i,1], nodes_world[i,1]],
            [anchors[i,2], nodes_world[i,2]], "g-", linewidth=2)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.legend()
ax.set_title("Cable Robot (Stiffness-Only, Final Pose)")
plt.show()

# ------------------------
# Cable length vs time (with natural lengths)
# ------------------------
plt.figure(figsize=(9,5))
for i in range(n):
    plt.plot(ts, cable_L[:, i], label=f"L{i+1} actual")
    plt.plot(ts, cable_L_nat[:, i], '--', alpha=0.6, label=f"L{i+1} nat")
plt.xlabel("Time [s]")
plt.ylabel("Cable Length [m]")
plt.title("Cable Lengths vs Time (with recalculated natural lengths)")
plt.legend(ncol=2, fontsize=8)
plt.grid(True, alpha=0.3)
plt.show()
