#!/usr/bin/env python3
"""
Generate a Pose Profile CSV:
t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg

Helix about Z axis:
- z from -50 to +50 mm
- radius 50 mm
- pitch 25 mm/turn => 4 turns over 100 mm

Includes:
- 1s linear move from (0,0,0) to helix start (R,0,z_start)
- helix segment
- 1s linear move back to (0,0,0)
"""

import os
import csv
import math


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def clamp01(u: float) -> float:
    return 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)


def generate_helix_pose_profile(
    out_path: str,
    rate_hz: float = 250.0,
    move_time_s: float = 1.0,
    helix_time_s: float = 1.0,   # adjust this to change speed
    radius_mm: float = 75.0,
    z_start_mm: float = -50.0,
    z_end_mm: float = 50.0,
    pitch_mm_per_turn: float = 25.0,
):
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    if move_time_s < 0 or helix_time_s < 0:
        raise ValueError("durations must be >= 0")
    if pitch_mm_per_turn == 0:
        raise ValueError("pitch_mm_per_turn must be nonzero")

    dz = z_end_mm - z_start_mm
    turns = dz / pitch_mm_per_turn  # 100/25 = 4 turns
    theta_total = 2.0 * math.pi * turns

    # Segment endpoints
    p0 = (0.0, 0.0, 0.0)
    p1 = (radius_mm, 0.0, z_start_mm)  # helix start
    p2 = (radius_mm, 0.0, z_end_mm)    # helix end (angle returns to 0 after integer turns)
    p3 = (0.0, 0.0, 0.0)

    total_time = move_time_s + helix_time_s + move_time_s
    dt = 1.0 / rate_hz
    n = int(math.floor(total_time / dt)) + 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"])

        for k in range(n):
            t = k * dt

            # Segment 1: linear move to helix start
            if t <= move_time_s and move_time_s > 0:
                u = clamp01(t / move_time_s)
                x = lerp(p0[0], p1[0], u)
                y = lerp(p0[1], p1[1], u)
                z = lerp(p0[2], p1[2], u)

            # Segment 2: helix
            elif t <= move_time_s + helix_time_s and helix_time_s > 0:
                th = t - move_time_s
                u = clamp01(th / helix_time_s)  # 0..1 along helix
                theta = u * theta_total
                x = radius_mm * math.cos(theta)
                y = radius_mm * math.sin(theta)
                z = lerp(z_start_mm, z_end_mm, u)

            # Segment 3: linear move back to origin
            else:
                th = t - (move_time_s + helix_time_s)
                if move_time_s > 0:
                    u = clamp01(th / move_time_s)
                else:
                    u = 1.0
                x = lerp(p2[0], p3[0], u)
                y = lerp(p2[1], p3[1], u)
                z = lerp(p2[2], p3[2], u)

            # Angles (deg) — keep flat for now
            roll = 0.0
            pitch = 0.0
            yaw = 0.0

            w.writerow([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{roll:.6f}", f"{pitch:.6f}", f"{yaw:.6f}"])

    print(f"Wrote pose profile: {out_path}")
    print(f"  rate_hz={rate_hz}, total_time={total_time:.3f}s, samples={n}")
    print(f"  helix: R={radius_mm}mm, z={z_start_mm}->{z_end_mm}mm, pitch={pitch_mm_per_turn}mm/turn, turns={turns:.3f}")


if __name__ == "__main__":
    # Put it directly into your GUI's Profiles folder if you run from the Controller directory
    out_csv = os.path.join("Profiles", "pose_helix_z_-50_50_R50_pitch25.csv")

    generate_helix_pose_profile(
        out_path=out_csv,
        rate_hz=250.0,
        move_time_s=1.0,
        helix_time_s=1.0,         # change speed here
        radius_mm=75.0,
        z_start_mm=-50.0,
        z_end_mm=50.0,
        pitch_mm_per_turn=25.0,
    )
