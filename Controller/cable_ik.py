# cable_ik.py

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z)

def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def v_norm(a: Vec3) -> float:
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def q_norm(q: Quat) -> Quat:
    w,x,y,z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w/n, x/n, y/n, z/n)

def q_mul(a: Quat, b: Quat) -> Quat:
    aw,ax,ay,az = a
    bw,bx,by,bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )

def q_from_axis_angle(axis: Vec3, angle_rad: float) -> Quat:
    ax, ay, az = axis
    n = math.sqrt(ax*ax + ay*ay + az*az)
    if n <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay, az = ax/n, ay/n, az/n
    s = math.sin(angle_rad/2.0)
    return q_norm((math.cos(angle_rad/2.0), ax*s, ay*s, az*s))

def q_to_R(q: Quat) -> Tuple[Tuple[float,float,float],Tuple[float,float,float],Tuple[float,float,float]]:
    # Rotation matrix from unit quaternion (w,x,y,z)
    w,x,y,z = q_norm(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return (
        (1.0 - 2.0*(yy+zz), 2.0*(xy - wz),     2.0*(xz + wy)),
        (2.0*(xy + wz),     1.0 - 2.0*(xx+zz), 2.0*(yz - wx)),
        (2.0*(xz - wy),     2.0*(yz + wx),     1.0 - 2.0*(xx+yy)),
    )

def R_mul_v(R, v: Vec3) -> Vec3:
    return (
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    )

@dataclass
class CableRobotGeometry:
    """
    Geometry definition for 6-cable platform.
    - anchors_world[i]: fixed anchor point of cable i in WORLD frame (meters)
    - attach_platform[i]: attachment point of cable i on the platform, in PLATFORM (hand) frame (meters)
    """
    anchors_world = [
        (+0.50, +0.50, +1.20),  # front-right-top
        (+0.50, -0.50, +1.20),  # rear-right-top
        (-0.50, +0.50, +1.20),  # front-left-top
        (-0.50, -0.50, +1.20),  # rear-left-top
        (+0.60, 0.00, +1.10),  # right-mid
        (-0.60, 0.00, +1.10),  # left-mid
    ]

    attach_platform = [
        (+0.50, +0.50, +1.20),  # front-right-top
        (+0.50, -0.50, +1.20),  # rear-right-top
        (-0.50, +0.50, +1.20),  # front-left-top
        (-0.50, -0.50, +1.20),  # rear-left-top
        (+0.60, 0.00, +1.10),  # right-mid
        (-0.60, 0.00, +1.10),  # left-mid
    ]

    def __post_init__(self):
        if len(self.anchors_world) != 6 or len(self.attach_platform) != 6:
            raise ValueError("Expected exactly 6 anchors and 6 attachment points.")

@dataclass
class WinchCalibration:
    """
    Converts cable length (m) to ODrive 'turns' for each axis.
    - spool_radius_m[i]: effective spool radius for axis i (meters)
    - gear_ratio[i]: motor_turns / spool_turn (use 1.0 if axis units are spool turns already)
    - sign[i]: +1 or -1 depending on motor direction for shortening cable with +turns
    - zero_length_m[i]: cable length (m) that corresponds to axis_turns=0.0 (your HOME reference)
    """
    spool_radius_m: List[float]
    gear_ratio: List[float]
    sign: List[float]
    zero_length_m: List[float]

    def __post_init__(self):
        for name, arr in [
            ("spool_radius_m", self.spool_radius_m),
            ("gear_ratio", self.gear_ratio),
            ("sign", self.sign),
            ("zero_length_m", self.zero_length_m),
        ]:
            if len(arr) != 6:
                raise ValueError(f"{name} must have length 6.")

def pose_to_cable_lengths_m(geom: CableRobotGeometry, t_world: Vec3, q_world: Quat) -> List[float]:
    """
    IK core: pose -> cable lengths (meters)
    L_i = || anchor_i - (t + R(q)*attach_i) ||
    """
    R = q_to_R(q_world)
    lengths = []
    for i in range(6):
        p_world = v_add(t_world, R_mul_v(R, geom.attach_platform[i]))
        d = v_sub(geom.anchors_world[i], p_world)
        lengths.append(v_norm(d))
    return lengths

def cable_lengths_to_turns(lengths_m: List[float], cal: WinchCalibration) -> List[float]:
    """
    Convert length (m) to axis turns.
    delta_turns = (L - L0) / (2*pi*r) * gear_ratio
    axis_turns = sign * delta_turns
    """
    turns = []
    for i in range(6):
        r = float(cal.spool_radius_m[i])
        if r <= 0.0:
            raise ValueError(f"spool_radius_m[{i}] must be > 0")
        L0 = float(cal.zero_length_m[i])
        dL = float(lengths_m[i]) - L0
        spool_turns = dL / (2.0 * math.pi * r)
        motor_turns = spool_turns * float(cal.gear_ratio[i])
        turns.append(float(cal.sign[i]) * motor_turns)
    return turns

def pose_to_axis_turns(
    geom: CableRobotGeometry,
    cal: WinchCalibration,
    t_world: Vec3,
    q_world: Quat,
) -> Tuple[List[float], List[float]]:
    lengths = pose_to_cable_lengths_m(geom, t_world, q_world)
    turns = cable_lengths_to_turns(lengths, cal)
    return turns, lengths
