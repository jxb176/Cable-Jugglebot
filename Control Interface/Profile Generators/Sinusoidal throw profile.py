#!/usr/bin/env python3
"""
Generate a Pose Profile CSV:
t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg

Z-only profile:
- Start at (0,0,0)
- 1s linear move down to z_min
- half-sinusoid (half cosine) up to z_max over sine_time_s
- 1s linear move back to z=0
"""

import os
import csv
import math


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def clamp01(u: float) -> float:
    return 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)


def half_sine_z(z_min: float, z_max: float, u01: float) -> float:
    """
    Smooth half-cycle from z_min to z_max.
    u01 in [0,1]. Uses half-cosine so dz/dt=0 at endpoints.
    """
    u = clamp01(u01)
    return z_min + (z_max - z_min) * 0.5 * (1.0 - math.cos(math.pi * u))


def generate_z_half_sine_pose_profile(
    out_path: str,
    rate_hz: float = 250.0,
    move_down_time_s: float = 1.0,
    sine_time_s: float = 4.0,
    move_back_time_s: float = 1.0,
    z_min_mm: float = -50.0,
    z_max_mm: float = 50.0,
):
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    if move_down_time_s < 0 or sine_time_s < 0 or move_back_time_s < 0:
        raise ValueError("durations must be >= 0")

    dt = 1.0 / rate_hz
    total_time = move_down_time_s + sine_time_s + move_back_time_s
    n = int(math.floor(total_time / dt)) + 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"])

        for k in range(n):
            t = k * dt

            x = 0.0
            y = 0.0

            # Segment 1: linear down 0 -> z_min
            if t <= move_down_time_s and move_down_time_s > 0:
                u = clamp01(t / move_down_time_s)
                z = lerp(0.0, z_min_mm, u)

            # Segment 2: half sinusoid up z_min -> z_max
            elif t <= move_down_time_s + sine_time_s and sine_time_s > 0:
                th = t - move_down_time_s
                u = clamp01(th / sine_time_s)
                z = half_sine_z(z_min_mm, z_max_mm, u)

            # Segment 3: linear back z_max -> 0
            else:
                th = t - (move_down_time_s + sine_time_s)
                if move_back_time_s > 0:
                    u = clamp01(th / move_back_time_s)
                else:
                    u = 1.0
                z = lerp(z_max_mm, 0.0, u)

            roll = 0.0
            pitch = 0.0
            yaw = 0.0

            w.writerow([
                f"{t:.6f}",
                f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                f"{roll:.6f}", f"{pitch:.6f}", f"{yaw:.6f}",
            ])

    print(f"Wrote pose profile: {out_path}")
    print(f"  rate_hz={rate_hz}, total_time={total_time:.3f}s, samples={n}")
    print(f"  z: 0 -> {z_min_mm} (linear {move_down_time_s}s) -> {z_max_mm} (half-sine {sine_time_s}s) -> 0 (linear {move_back_time_s}s)")


if __name__ == "__main__":
    out_csv = os.path.join("Profiles", "pose_z_half_sine_zmin-50_zmax50.csv")
    generate_z_half_sine_pose_profile(
        out_path=out_csv,
        rate_hz=250.0,
        move_down_time_s=0.25,
        sine_time_s=0.05,
        move_back_time_s=0.25,
        z_min_mm=-100.0,
        z_max_mm=100.0,
    )
