#!/usr/bin/env python3
"""
view_sim_log.py

Usage:
  python view_sim_log.py --npz sim_log.npz
  python view_sim_log.py --npz sim_log.npz --csv sim_log.csv
  python view_sim_log.py --npz sim_log.npz --no-plots

What it expects inside the npz:
  t    : (N,)
  q    : (N,5)
  qref : (N,5)   (optional but recommended)
  T    : (N,6)   (optional)

Plots:
  - q vs qref for x,y,z,roll,pitch
  - tensions vs time
"""

import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="sim_log.npz", help="Path to sim_log.npz")
    ap.add_argument("--csv", default=None, help="Optional CSV export path")
    ap.add_argument("--no-plots", action="store_true", help="Do not show plots")
    args = ap.parse_args()

    log = np.load(args.npz)

    print("\n=== sim_log.npz contents ===")
    print("keys:", log.files)
    for k in log.files:
        arr = log[k]
        print(f"  {k:6s} shape={arr.shape} dtype={arr.dtype}")

    # Required
    if "t" not in log.files or "q" not in log.files:
        raise ValueError("NPZ must contain at least 't' and 'q' arrays.")

    t = log["t"]
    q = log["q"]
    qd = log["qd"] if "qd" in log.files else None
    qdref = log["qdref"] if "qdref" in log.files else None
    qdd = log["qdd"] if "qdd" in log.files else None
    qddref = log["qddref"] if "qddref" in log.files else None
    tau_ff = log["tau_ff"] if "tau_ff" in log.files else None
    tau_fb = log["tau_fb"] if "tau_fb" in log.files else None
    tau_des = log["tau_des"] if "tau_des" in log.files else None

    qref = log["qref"] if "qref" in log.files else None
    T = log["T"] if "T" in log.files else None

    # Basic sanity stats
    print("\n=== sanity ===")
    print("t:  ", float(t[0]), "->", float(t[-1]), f"(N={len(t)})")
    print("NaNs in q:", bool(np.isnan(q).any()))
    if qref is not None:
        print("NaNs in qref:", bool(np.isnan(qref).any()))
    if T is not None:
        print("NaNs in T:", bool(np.isnan(T).any()))
    if tau_ff is not None:
        print("NaNs in tau_ff:", bool(np.isnan(tau_ff).any()))
    if tau_fb is not None:
        print("NaNs in tau_fb:", bool(np.isnan(tau_fb).any()))
    if tau_des is not None:
        print("NaNs in tau_des:", bool(np.isnan(tau_des).any()))

    if qref is not None:
        err = q - qref
        max_abs = np.max(np.abs(err), axis=0)
        rms = np.sqrt(np.mean(err**2, axis=0))
        print("\n=== tracking error (q - qref) ===")
        labels = ["x [m]", "y [m]", "z [m]", "roll [rad]", "pitch [rad]"]
        for i, lab in enumerate(labels):
            print(f"{lab:10s}  max|e|={max_abs[i]: .6g}   rms={rms[i]: .6g}")

    if T is not None:
        print("\n=== tension stats ===")
        Tmin = T.min(axis=0)
        Tmax = T.max(axis=0)
        for i in range(T.shape[1]):
            print(f"cable {i+1}:  min={Tmin[i]: .3f}  max={Tmax[i]: .3f}")

    # Optional CSV export
    if args.csv is not None:
        import pandas as pd
        cols = ["t"]
        data = [t.reshape(-1, 1)]

        # q
        q_cols = ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad"]
        cols += q_cols
        data.append(q)

        # qref
        if qref is not None:
            qref_cols = ["xref_m", "yref_m", "zref_m", "rollref_rad", "pitchref_rad"]
            cols += qref_cols
            data.append(qref)

        # tensions
        if T is not None:
            T_cols = [f"T{i+1}_N" for i in range(T.shape[1])]
            cols += T_cols
            data.append(T)

        mat = np.hstack(data)
        df = pd.DataFrame(mat, columns=cols)
        df.to_csv(args.csv, index=False)
        print(f"\nWrote CSV: {args.csv}")

    # Plots
    if not args.no_plots:
        import matplotlib.pyplot as plt

        labels = ["x [m]", "y [m]", "z [m]", "roll [rad]", "pitch [rad]"]

        # q vs qref
        plt.figure(figsize=(9, 8))
        for i in range(min(5, q.shape[1])):
            ax = plt.subplot(5, 1, i + 1)
            ax.plot(t, q[:, i], label="actual")
            if qref is not None:
                ax.plot(t, qref[:, i], "--", label="ref")
            ax.set_ylabel(labels[i])
            ax.grid(True)
            if i == 0:
                ax.legend()
            if i == 4:
                ax.set_xlabel("time [s]")
        plt.tight_layout()

        # tensions
        if T is not None:
            plt.figure(figsize=(9, 4))
            for i in range(T.shape[1]):
                plt.plot(t, T[:, i], label=f"cable {i+1}")
            plt.ylabel("tension [N]")
            plt.xlabel("time [s]")
            plt.grid(True)
            plt.legend(ncol=3)
            plt.tight_layout()

        have_all = (qref is not None) and (qd is not None) and (qdref is not None) and (qdd is not None) and (
                    qddref is not None)

        # z axis pos, vel, acc
        if have_all:
            plt.figure(figsize=(9, 7))

            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(t, q[:, 2], label="actual")
            ax1.plot(t, qref[:, 2], "--", label="ref")
            ax1.set_ylabel("z [m]")
            ax1.grid(True)
            ax1.legend()

            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(t, qd[:, 2], label="actual")
            ax2.plot(t, qdref[:, 2], "--", label="ref")
            ax2.set_ylabel("vz [m/s]")
            ax2.grid(True)
            ax2.legend()

            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(t, qdd[:, 2], label="actual")
            ax3.plot(t, qddref[:, 2], "--", label="ref")
            ax3.set_ylabel("az [m/s²]")
            ax3.set_xlabel("time [s]")
            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
        else:
            print("Note: skipping z pos/vel/acc plot (need qref, qd, qdref, qdd, qddref)")

        # --- NEW: Z generalized force/torque components (ff vs fb vs total) ---
        if (tau_ff is not None) and (tau_fb is not None) and (tau_des is not None):
            import matplotlib.pyplot as plt

            plt.figure(figsize=(9, 6))

            # Index 2 corresponds to z in your [x,y,z,roll,pitch] ordering
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(t, tau_ff[:, 2], label="tau_ff")
            ax1.set_ylabel("tau_ff,z")
            ax1.grid(True)
            ax1.legend()

            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(t, tau_fb[:, 2], label="tau_fb")
            ax2.set_ylabel("tau_fb,z")
            ax2.grid(True)
            ax2.legend()

            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(t, tau_des[:, 2], label="tau_des")
            ax3.set_ylabel("tau_des,z")
            ax3.set_xlabel("time [s]")
            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
        else:
            print("Note: tau_ff/tau_fb/tau_des not found; skipping wrench plots.")

        # --- OPTIONAL: tau_ff/tau_fb/tau_des for all 5 dofs ---
        if (tau_ff is not None) and (tau_fb is not None) and (tau_des is not None):
            labels_tau = ["x", "y", "z", "roll", "pitch"]
            plt.figure(figsize=(9, 8))
            for i in range(5):
                ax = plt.subplot(5, 1, i + 1)
                ax.plot(t, tau_des[:, i], label="tau_des")
                ax.plot(t, tau_ff[:, i], "--", label="tau_ff")
                ax.plot(t, tau_fb[:, i], ":", label="tau_fb")
                ax.set_ylabel(labels_tau[i])
                ax.grid(True)
                if i == 0:
                    ax.legend()
                if i == 4:
                    ax.set_xlabel("time [s]")
            plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()
