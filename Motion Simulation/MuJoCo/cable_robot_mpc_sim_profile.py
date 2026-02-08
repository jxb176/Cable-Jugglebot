#!/usr/bin/env python3
"""
cable_robot_mpc_sim_profile.py

Stable simulation loop for a 6-cable, 5-DOF platform (x,y,z,roll,pitch; yaw fixed).

Implements:
- Allocation using MuJoCo tendon Jacobians (no analytical wrench map).
- Gravity/bias compensation via data.qfrc_bias.
- Pretension + explicit clamping of cable tensions (ctrl).
- Reference tracking from a CSV pose command (t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg).

Notes:
- Tendon motor ctrl is interpreted as force along the tendon. With gear="-1" in XML,
  positive ctrl should pull (tension).
- This controller tracks generalized coordinates (q) directly for now; later you can
  switch the outer loop to EE pose error and an MPC/QP formulation.
"""
import argparse
import numpy as np

import mujoco
from mujoco import viewer

def get_mass_matrix_dofs(model, data, dof_idxs):
    """
    Return the joint-space mass matrix for the selected DOFs as a dense (ndof x ndof).
    """
    # full nv x nv mass matrix
    M_full = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, M_full, data.qM)

    # extract submatrix for our controlled dofs
    M = M_full[np.ix_(dof_idxs, dof_idxs)]
    return M

def tendon_jacobian_fd(model, data, tendon_ids, qpos_adr, eps_pos=1e-6, eps_ang=1e-6):
    """
    Compute J = d(tendon_length)/d(q_dof) with finite differences.

    tendon_ids: list of tendon ids for cable1..cable6
    qpos_adr: indices into data.qpos for [jx, jy, jz, jroll, jpitch]
    Returns J shape (ntendon, ndof) = (6, 5)
    """
    nt = len(tendon_ids)
    nd = len(qpos_adr)
    J = np.zeros((nt, nd), dtype=float)

    # Save state
    qpos0 = data.qpos.copy()

    # Make sure tendon lengths are current
    mujoco.mj_forward(model, data)
    l0 = data.ten_length.copy()

    for j in range(nd):
        idx = int(qpos_adr[j])
        is_angle = (j >= 3)  # roll, pitch
        eps = eps_ang if is_angle else eps_pos

        # +eps
        data.qpos[:] = qpos0
        data.qpos[idx] = qpos0[idx] + eps
        mujoco.mj_forward(model, data)
        lp = data.ten_length.copy()

        # -eps
        data.qpos[:] = qpos0
        data.qpos[idx] = qpos0[idx] - eps
        mujoco.mj_forward(model, data)
        lm = data.ten_length.copy()

        dl_dq = (lp - lm) / (2.0 * eps)

        # pull out only the tendons we care about, in cable order
        for i, tid in enumerate(tendon_ids):
            J[i, j] = dl_dq[tid]

    # Restore
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    return J



def solve_tensions_least_squares(J, tau_des, T0, Tmax, lam=1e-3, iters=80):
    """Solve for tensions T in [0, Tmax] that best match generalized forces.

    We use projected gradient descent on:
        min_T || (-J^T) T - tau_des ||^2 + lam ||T - T0||^2
        s.t. 0 <= T <= Tmax

    T0 acts as pretension (bias toward non-slack), but the bounds are always enforced.
    """
    # Map tensions -> generalized force
    A = -J.T                       # (ndof x ntendon)
    nt = A.shape[1]
    lb = np.zeros(nt)
    ub = np.full(nt, Tmax)

    # Initialize near pretension
    T = np.clip(np.full(nt, T0), lb, ub)

    # Quadratic objective:
    # f(T)=||A T - tau||^2 + lam ||T - T0||^2
    # grad = 2 A^T (A T - tau) + 2 lam (T - T0)
    AtA = A.T @ A
    Atb = A.T @ tau_des

    # Lipschitz estimate for step (safe)
    L = np.linalg.norm(2*AtA + 2*lam*np.eye(nt), ord=2)
    step = 1.0 / max(L, 1e-9)

    T0v = np.full(nt, T0)

    for _ in range(iters):
        grad = 2*(AtA @ T - Atb) + 2*lam*(T - T0v)
        T = np.clip(T - step * grad, lb, ub)
    return T


def load_pose_profile_csv(path):
    """
    Load pose command CSV.

    Supports two formats:
      1) pose-only:  t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, (yaw_deg optional)
         -> returns qref and qdref via finite differences, qddref zeros
      2) full: additionally includes vx_mps, vy_mps, vz_mps and ax_mps2, ay_mps2, az_mps2
         -> returns qref, qdref, qddref using provided columns (no differencing)
    Returns:
        t: (N,)
        q: (N,5)  [m, m, m, rad, rad]
        qd: (N,5) [m/s, ..., rad/s]
        qdd:(N,5) [m/s^2, ..., rad/s^2]
    """
    import pandas as pd
    df = pd.read_csv(path)

    required = ["t", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    # --- time sanitize (same as you already do) ---
    t = df["t"].to_numpy(dtype=float)
    order = np.argsort(t)
    df = df.iloc[order].reset_index(drop=True)
    t = df["t"].to_numpy(dtype=float)

    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.diff(t) > 1e-9
    df = df.loc[keep].reset_index(drop=True)
    t = df["t"].to_numpy(dtype=float)

    # --- positions / angles ---
    q = np.zeros((len(t), 5), dtype=float)
    q[:, 0] = df["x_mm"].to_numpy(dtype=float) / 1000.0
    q[:, 1] = df["y_mm"].to_numpy(dtype=float) / 1000.0
    q[:, 2] = df["z_mm"].to_numpy(dtype=float) / 1000.0

    roll = np.deg2rad(df["roll_deg"].to_numpy(dtype=float))
    pitch = np.deg2rad(df["pitch_deg"].to_numpy(dtype=float))
    q[:, 3] = np.unwrap(roll)
    q[:, 4] = np.unwrap(pitch)

    # --- velocities ---
    has_v = all(c in df.columns for c in ["vx_mps", "vy_mps", "vz_mps"])
    qd = np.zeros_like(q)
    if has_v:
        qd[:, 0] = df["vx_mps"].to_numpy(dtype=float)
        qd[:, 1] = df["vy_mps"].to_numpy(dtype=float)
        qd[:, 2] = df["vz_mps"].to_numpy(dtype=float)
        # roll/pitch rates not provided yet; keep zero
    else:
        # fallback: safe finite differences (your current method)
        for k in range(1, len(t) - 1):
            dt1 = t[k] - t[k - 1]
            dt2 = t[k + 1] - t[k]
            qd[k] = (q[k + 1] - q[k - 1]) / (dt1 + dt2)
        qd[0] = (q[1] - q[0]) / max(t[1] - t[0], 1e-9)
        qd[-1] = (q[-1] - q[-2]) / max(t[-1] - t[-2], 1e-9)

    # --- accelerations ---
    has_a = all(c in df.columns for c in ["ax_mps2", "ay_mps2", "az_mps2"])
    qdd = np.zeros_like(q)
    if has_a:
        qdd[:, 0] = df["ax_mps2"].to_numpy(dtype=float)
        qdd[:, 1] = df["ay_mps2"].to_numpy(dtype=float)
        qdd[:, 2] = df["az_mps2"].to_numpy(dtype=float)
        # roll/pitch accel not provided yet; keep zero
    else:
        # keep zero if not provided; we'll compute qddref inside sim only if you want
        pass

    return t, q, qd, qdd



def make_reference_from_profile(t_prof, q_prof, qd_prof, qdd_prof=None):
    """Return callable reference(t)->(qref, qdref, qddref) using linear interpolation."""
    def interp_vec(tt, arr):
        out = np.zeros(arr.shape[1], dtype=float)
        for j in range(arr.shape[1]):
            out[j] = np.interp(tt, t_prof, arr[:, j])
        return out

    if qdd_prof is None:
        qdd_prof = np.zeros_like(q_prof)

    def reference(t):
        tt = float(np.clip(t, t_prof[0], t_prof[-1]))
        return interp_vec(tt, q_prof), interp_vec(tt, qd_prof), interp_vec(tt, qdd_prof)

    return reference



def make_default_reference():
    """Fallback smooth reference if no CSV provided."""
    def reference(t):
        x = 0.05*np.sin(2*np.pi*0.15*t)
        y = 0.05*np.cos(2*np.pi*0.12*t)
        z = 0.10 + 0.03*np.sin(2*np.pi*0.10*t)
        roll  = np.deg2rad(5.0)*np.sin(2*np.pi*0.11*t)
        pitch = np.deg2rad(5.0)*np.cos(2*np.pi*0.09*t)
        qref = np.array([x, y, z, roll, pitch], dtype=float)
        # crude analytic velocity for this default (fine for demo)
        qdref = np.zeros(5, dtype=float)
        qddref = np.zeros(5, dtype=float)
        return qref, qdref, qddref
    return reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="cable_robot_5dof.xml", help="MuJoCo XML model path")
    ap.add_argument("--profile", default="pose_cmd.csv", help="CSV profile path (pose_cmd.csv)")
    ap.add_argument("--no-profile", action="store_true", help="Ignore CSV and use default sine reference")
    ap.add_argument("--T0", type=float, default=30.0, help="Pretension (N)")
    ap.add_argument("--Tmax", type=float, default=None, help="Max tension (N) override")
    ap.add_argument("--Kp", type=float, default=80.0, help="Base Kp for xyz (N/m) and angles (Nm/rad) scaling")
    ap.add_argument("--Kd", type=float, default=20.0, help="Base Kd for xyz (N/(m/s)) and angles (Nm/(rad/s)) scaling")
    ap.add_argument("--sim-time", type=float, default=10.0, help="Simulation duration (s)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # DOF indices: for simple joints, qpos order matches joints and qvel adr gives nv indices
    joint_names = ["jx", "jy", "jz", "jroll", "jpitch"]
    qpos_adr = []
    qvel_adr = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr.append(model.jnt_qposadr[jid])
        qvel_adr.append(model.jnt_dofadr[jid])
    qpos_adr = np.array(qpos_adr, dtype=int)
    dof_idxs = np.array(qvel_adr, dtype=int)  # indices into nv

    # Tendon ids in the same order as actuators/cables
    tendon_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, f"cable{i}") for i in range(1, 7)]
    assert all(tid >= 0 for tid in tendon_ids), "One or more tendons cable1..cable6 not found in XML"

    # Tension limits
    Tmax = args.Tmax
    if Tmax is None:
        try:
            Tmax = float(np.min(model.actuator_ctrlrange[:, 1]))
        except Exception:
            Tmax = 150.0

    # Reference
    if (not args.no_profile) and args.profile:
        t_prof, q_prof, qd_prof, qdd_prof = load_pose_profile_csv(args.profile)
        reference = make_reference_from_profile(t_prof, q_prof, qd_prof, qdd_prof)
        t_end = float(t_prof[-1])
        sim_time = min(args.sim_time, t_end)
    else:
        reference = make_default_reference()
        sim_time = args.sim_time

    # Gains (start conservative; adjust as needed)
    # xyz gains in N/m; angle gains in Nm/rad (scale down a bit by default)
    Kp = np.diag([args.Kp, args.Kp, args.Kp*1.2, args.Kp*0.15, args.Kp*0.15])
    Kd = np.diag([args.Kd, args.Kd, args.Kd*1.2, args.Kd*0.15, args.Kd*0.15])

    # Logs
    max_steps = int(sim_time / model.opt.timestep) + 1
    log_t = np.zeros(max_steps)
    log_q = np.zeros((max_steps, 5))
    log_qd = np.zeros((max_steps, 5))
    log_qdd = np.zeros((max_steps, 5))

    log_qref = np.zeros((max_steps, 5))
    log_qdref = np.zeros((max_steps, 5))
    log_qddref = np.zeros((max_steps, 5))

    log_tau_ff = np.zeros((max_steps, 5))
    log_tau_fb = np.zeros((max_steps, 5))
    log_tau_des = np.zeros((max_steps, 5))

    log_T = np.zeros((max_steps, 6))

    qd_prev = None
    qdref_prev = None
    t_prev = None

    # Viewer
    with viewer.launch_passive(model, data) as v:
        k = 0
        while data.time < sim_time - 1e-9:
            # Current generalized coordinates and velocities
            q = data.qpos[qpos_adr].copy()
            qd = data.qvel[dof_idxs].copy()

            # Reference
            qref, qdref, qddref = reference(data.time)

            # Acceleration
            if qd_prev is None:
                qdd = np.zeros_like(qd)
            else:
                dt = max(float(data.time - t_prev), 1e-9)
                qdd = (qd - qd_prev) / dt

            # Compute wrench

            e = qref - q
            ed = qdref - qd

            # Feedforward inverse dynamics term
            M = get_mass_matrix_dofs(model, data, dof_idxs)
            b = data.qfrc_bias[dof_idxs].copy()
            tau_ff = M @ qddref + b

            # Feedback stabilization
            tau_fb = (Kp @ e) + (Kd @ ed)

            tau_des = tau_ff + tau_fb

            # Tendon Jacobian and tension allocation
            J = tendon_jacobian_fd(model, data, tendon_ids, qpos_adr)
            T = solve_tensions_least_squares(J, tau_des, T0=args.T0, Tmax=Tmax, lam=1e-2, iters=100)

            # Explicit clamp (even though ctrllimited is set)
            T = np.clip(T, 0.0, Tmax)
            data.ctrl[:] = T

            # Step simulation
            mujoco.mj_step(model, data)

            # Log (note: log q/qd at the time associated with data.time BEFORE mj_step)
            log_t[k] = data.time
            log_q[k, :] = q
            log_qd[k, :] = qd
            log_qdd[k, :] = qdd

            log_qref[k, :] = qref
            log_qdref[k, :] = qdref
            log_qddref[k, :] = qddref

            log_tau_ff[k, :] = tau_ff
            log_tau_fb[k, :] = tau_fb
            log_tau_des[k, :] = tau_des

            log_T[k, :] = T

            # Update previous values for next acceleration estimate
            qd_prev = qd
            qdref_prev = qdref
            t_prev = data.time

            k += 1

            # Render at ~60 Hz
            if k % 16 == 0:
                v.sync()

    # Trim logs
    log_t = log_t[:k]
    log_q = log_q[:k]
    log_qref = log_qref[:k]
    log_qd = log_qd[:k]
    log_qdd = log_qdd[:k]
    log_qdref = log_qdref[:k]
    log_qddref = log_qddref[:k]
    log_tau_ff = log_tau_ff[:k]
    log_tau_fb = log_tau_fb[:k]
    log_tau_des = log_tau_des[:k]
    log_T = log_T[:k]

    np.savez(
        "sim_log.npz",
        t=log_t,
        q=log_q, qd=log_qd, qdd=log_qdd,
        qref=log_qref, qdref=log_qdref, qddref=log_qddref,
        T=log_T,
        tau_ff=log_tau_ff,
        tau_fb=log_tau_fb,
        tau_des=log_tau_des,
    )

    print("Saved sim_log.npz")


if __name__ == "__main__":
    main()
