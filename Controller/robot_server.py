# robot_server.py
import socket
import json
import time
import threading
import os
import logging
from datetime import datetime
import subprocess
import asyncio
from ODriveCANSimple import ODriveCanManager, AxisState  # <-- your new module
from typing import List, Tuple

TCP_CMD_PORT = 5555
UDP_TELEM_PORT = 5556

# -------- ODrive CAN configuration --------
ODRIVE_INTERFACE = "can0"
ODRIVE_BITRATE = 1_000_000  # 1 Mbps
AXIS_NODE_IDS = [0, 1, 2, 3, 4, 5]
ODRIVE_COMMAND_RATE_HZ = 500.0
ODRIVE_LOG_RATE_HZ = 2.0
TELEMETRY_RATE_HZ = 50.0

# -------- Capstan / units configuration --------
# +turns (ODrive) reduces cable length, so use negative mm/turn such that +mm command extends cable.
MM_PER_TURN = [-62.832] * 6  # 2*pi*10mm = 62.832 mm/turn, with sign convention applied
# Pretension mapping: tension [N] -> capstan torque [Nm]
CAPSTAN_RADIUS_M = 0.010  # 10 mm
TORQUE_PER_TENSION = CAPSTAN_RADIUS_M  # Nm per N  (T = F*r)

#CLEANUP into ODRIVE library
# ODrive controller modes (CANSimple Set_Controller_Mode)
CONTROL_MODE_TORQUE = 1      # aka "CurrentControl" in some docs
CONTROL_MODE_POSITION = 3
INPUT_MODE_PASSTHROUGH = 1

def mm_to_turns(mm_list):
    """Convert [mm] -> [turns] elementwise using MM_PER_TURN."""
    if not isinstance(mm_list, (list, tuple)) or len(mm_list) != 6:
        raise ValueError("mm_list must be length-6 list/tuple")
    out = []
    for i in range(6):
        mm = float(mm_list[i])
        k = float(MM_PER_TURN[i])  # mm/turn
        out.append(mm / k)         # turns
    return out

def turns_to_mm(turns_list):
    """Convert [turns] -> [mm] elementwise using MM_PER_TURN."""
    if not isinstance(turns_list, (list, tuple)) or len(turns_list) != 6:
        raise ValueError("turns_list must be length-6 list/tuple")
    out = []
    for i in range(6):
        trn = float(turns_list[i])
        k = float(MM_PER_TURN[i])  # mm/turn
        out.append(trn * k)        # mm
    return out

def _coerce_vec6_to_mm(msg, field_name: str):
    vec = msg.get(field_name, [])
    units = (msg.get("units") or "mm").lower()
    if not isinstance(vec, (list, tuple)) or len(vec) != 6:
        raise ValueError(f"{field_name} must be length-6 list")
    vec = [float(x) for x in vec]

    if units == "mm":
        return vec
    elif units == "turns":
        return turns_to_mm(vec)
    else:
        raise ValueError(f"Unknown units '{units}' (expected 'mm' or 'turns')")

os.environ.setdefault("CAN_CHANNEL", ODRIVE_INTERFACE)
os.environ.setdefault("CAN_BITRATE", str(ODRIVE_BITRATE))

try:
    import odrive_can as odc
except Exception:
    odc = None

# -------- Logging setup --------
def _init_logging():
    logs_dir = os.path.join(os.getcwd(), "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"robot_{ts}.log")
    logger = logging.getLogger("robot")
    logger.setLevel(logging.DEBUG)   #INFO for low level, Set to DEBUG for more verbose logging
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path

logger, LOG_FILE_PATH = _init_logging()


def ensure_can_interface_up(ifname: str, bitrate: int) -> bool:
    try:
        res = subprocess.run(
            ["ip", "link", "show", ifname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if res.returncode == 0:
            out = res.stdout.lower()
            if " state up " in out or "<up," in out or "up>" in out:
                logger.info(f"[CAN] Interface {ifname} already UP")
                return True
        else:
            logger.warning(f"[CAN] '{ifname}' not found: {res.stderr.strip()}")

        logger.info(f"[CAN] Bringing up {ifname} @ {bitrate} bps")
        subprocess.run(["ip", "link", "set", ifname, "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cfg = subprocess.run(
            ["ip", "link", "set", ifname, "type", "can", "bitrate", str(bitrate)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if cfg.returncode != 0:
            logger.error(f"[CAN] Failed to configure {ifname}: {cfg.stderr.strip()}")
            return False
        up = subprocess.run(
            ["ip", "link", "set", ifname, "up"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if up.returncode != 0:
            logger.error(f"[CAN] Failed to bring {ifname} up: {up.stderr.strip()}")
            return False
        return True
    except Exception as e:
        logger.error(f"[CAN] Error: {e}")
        return False



class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.controller_ip = None
        self.axes_pos_cmd = [0.0] * 6
        self.state = "disable"
        self.state_version = 0
        self.home_pos = [0.0] * 6
        self.home_version = 0
        self.profile = []
        self.player_thread = None
        self.axes_pos_estimate = [None] * 6
        self.axes_vel_estimate = [None] * 6
        self.axes_bus_voltage = [None] * 6
        self.axes_bus_current = [None] * 6
        self.axes_motor_current = [None] * 6
        self.axes_temp_fet = [None] * 6
        self.axes_temp_motor = [None] * 6
        self.axes_axis_error = [None] * 6
        self.axes_axis_state = [None] * 6
        self.axes_proc_result = [None] * 6
        self.telem_thread = None
        self.telem_stop = threading.Event()
        self.pret_upper_N = 0.0
        self.pret_lower_N = 0.0
        self.pret_version = 0

    def set_controller_ip(self, ip):
        with self.lock:
            self.controller_ip = ip
        logger.info(f"Controller IP set to {ip}")

    def get_controller_ip(self):
        with self.lock:
            return self.controller_ip

    def get_pos_cmd(self):
        with self.lock:
            return list(self.axes_pos_cmd)

    def get_pos_fbk(self):
        with self.lock:
            return list(self.axes_pos_estimate)

    def get_vel_fbk(self):
        with self.lock:
            return list(self.axes_vel_estimate)

    #Home methods
    def request_home(self, home_pos):
        if not isinstance(home_pos, (list, tuple)) or len(home_pos) != 6:
            raise ValueError("home_pos must be length-6 list/tuple")
        with self.lock:
            self.home_pos = [float(x) for x in home_pos]
            self.home_version += 1
        logger.info("HOME requested (mm): " + ", ".join(f"{x:.3f}" for x in self.home_pos))

    def get_home_version(self):
        with self.lock:
            return self.home_version

    def get_home_pos(self):
        with self.lock:
            return list(self.home_pos)

    #Set methods
    def set_axis_feedback(
            self,
            axis_id: int,
            pos_estimate=None,
            vel_estimate=None,
            bus_voltage=None,
            bus_current=None,
            motor_current=None,
            temp_fet=None,
            temp_motor=None,
            axis_error=None,
            axis_state=None,
            proc_result=None
    ):
        """Store measured feedback for a single axis index (0..5)."""
        if not (0 <= int(axis_id) < 6):                         #This hardcodes id's 0-5, this should be upgraded to check against a list of initialized controllers
            return
        with self.lock:
            if pos_estimate is not None:
                try:
                    self.axes_pos_estimate[axis_id] = float(pos_estimate)
                except Exception:
                    pass
            if vel_estimate is not None:
                try:
                    self.axes_vel_estimate[axis_id] = float(vel_estimate)
                except Exception:
                    pass
            if bus_voltage is not None:
                try:
                    self.axes_bus_voltage[axis_id] = float(bus_voltage)
                except Exception:
                    pass
            if bus_current is not None:  # <-- new
                try:
                    self.axes_bus_current[axis_id] = float(bus_current)
                except Exception:
                    pass
            if motor_current is not None:
                try:
                    self.axes_motor_current[axis_id] = float(motor_current)
                except Exception:
                    pass
            if temp_fet is not None:
                try:
                    self.axes_temp_fet[axis_id] = float(temp_fet)
                except Exception:
                    pass
            if temp_motor is not None:
                try:
                    self.axes_temp_motor[axis_id] = float(temp_motor)
                except Exception:
                    pass
            if axis_error is not None:
                try:
                    self.axes_axis_error[axis_id] = int(axis_error)
                except Exception:
                    pass
            if axis_state is not None:
                try:
                    self.axes_axis_state[axis_id] = int(axis_state)
                except Exception:
                    pass
            if proc_result is not None:
                try:
                    self.axes_proc_result[axis_id] = int(proc_result)
                except Exception:
                    pass

    def get_bus_voltage(self):
        """Return list of bus voltage values (may contain None)."""
        with self.lock:
            return list(self.axes_bus_voltage)

    def get_bus_current(self):
        with self.lock:
            return list(self.axes_bus_current)

    def get_motor_current(self):
        with self.lock:
            return list(self.axes_motor_current)

    def get_temp_fet(self):
        with self.lock:
            return list(self.axes_temp_fet)

    def get_temp_motor(self):
        with self.lock:
            return list(self.axes_temp_motor)

    def get_axis_error(self):
        with self.lock:
            return list(self.axes_axis_error)

    def get_axis_state(self):
        with self.lock:
            return list(self.axes_axis_state)

    def get_proc_result(self):
        with self.lock:
            return list(self.axes_proc_result)

    def set_axes(self, positions):
        if not isinstance(positions, (list, tuple)) or len(positions) != 6:
            raise ValueError("positions must be length-6 list/tuple")
        with self.lock:
            self.axes_pos_cmd = [float(x) for x in positions]
        logger.info("Axes target set (mm): " + ", ".join(f"{x:.3f}" for x in self.axes_pos_cmd))

    def set_state(self, value: str):
        value = str(value).lower()
        if value not in ("enable", "disable", "estop", "pretension"):
            raise ValueError("invalid state")
        with self.lock:
            self.state = value
            self.state_version += 1
        logger.info(f"State set to: {value} (version {self.state_version})")

    def get_state(self):
        with self.lock:
            return self.state

    def get_state_version(self):
        with self.lock:
            return self.state_version

    def set_profile(self, profile_points):
        if not isinstance(profile_points, (list, tuple)) or len(profile_points) == 0:
            raise ValueError("profile must be a non-empty list")
        prof = []
        for row in profile_points:
            if not isinstance(row, (list, tuple)) or len(row) < 7:
                raise ValueError("each profile row must be [t, a1..a6]")
            t = float(row[0])
            axes = [float(x) for x in row[1:7]]
            prof.append((t, axes))
        times = [p[0] for p in prof]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("profile time column must be non-decreasing")
        with self.lock:
            self.profile = prof
        logger.info(f"Profile uploaded: {len(prof)} points, duration {prof[-1][0] - prof[0][0]:.3f}s")

    def get_profile(self):
        """Return the currently stored profile as a list of (t, axes)."""
        with self.lock:
            return list(self.profile)

    def start_profile(self, rate_hz: float):
        """Start executing the uploaded profile at a given rate (Hz)."""
        self.stop_profile()
        prof = self.get_profile()
        if not prof:
            raise RuntimeError("no profile uploaded")
        player = ProfilePlayer(self, prof, rate_hz)
        with self.lock:
            self.player_thread = player
        logger.info(f"Profile start at {rate_hz:.1f} Hz")
        player.start()

    def stop_profile(self):
        """Stop any running profile playback."""
        with self.lock:
            player = self.player_thread
            self.player_thread = None
        if player and player.is_alive():
            player.stop()
            player.join(timeout=1.0)
            logger.info("Profile stopped")

    # --- Telemetry lifecycle ---
    def start_telem(self, udp_sock, controller_addr):
        self.stop_telem()
        self.telem_stop.clear()
        t = threading.Thread(
            target=udp_telemetry_sender,
            args=(self, udp_sock, self.telem_stop),
            daemon=True,
        )
        self.telem_thread = t
        t.start()
        logger.info("[UDP] Telemetry thread started")

    def stop_telem(self):
        if self.telem_thread and self.telem_thread.is_alive():
            self.telem_stop.set()
            self.telem_thread.join(timeout=1.0)
            logger.info("[UDP] Telemetry thread stopped")
        self.telem_thread = None

    #Pretension methods
    def request_pretension(self, upper_N: float, lower_N: float):
        with self.lock:
            self.pret_upper_N = float(upper_N)
            self.pret_lower_N = float(lower_N)
            self.pret_version += 1
            self.state = "pretension"
            self.state_version += 1
        logger.info(f"[PRET] requested upper={self.pret_upper_N:.3f} N, lower={self.pret_lower_N:.3f} N "
                    f"(pret_version {self.pret_version})")

    def get_pretension(self):
        with self.lock:
            return float(self.pret_upper_N), float(self.pret_lower_N)

    def get_pretension_version(self):
        with self.lock:
            return int(self.pret_version)


class ProfilePlayer(threading.Thread):
    """Plays a time-position profile with linear interpolation at fixed rate."""

    def __init__(self, state: RobotState, profile: list[tuple[float, list[float]]], rate_hz: float):
        super().__init__(daemon=True)
        self.state = state
        self._stop = threading.Event()
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.dt = 1.0 / rate_hz

        # Normalize profile times so playback starts at t=0
        if not profile:
            raise ValueError("empty profile")
        t0 = float(profile[0][0])
        norm = []
        for t, axes in profile:
            norm.append((float(t) - t0, [float(x) for x in axes]))
        self.norm_profile = norm
        self.duration = norm[-1][0] if norm else 0.0

    def stop(self):
        self._stop.set()

    def run(self):
        if self.duration <= 0.0:
            # immediate set and exit
            self.state.set_axes(self.norm_profile[-1][1])
            logger.info("[PROFILE] Zero-duration profile applied")
            return

        logger.info(f"[PROFILE] Starting playback at {1.0/self.dt:.1f} Hz, duration {self.duration:.3f}s")
        start = time.perf_counter()
        k = 0
        while not self._stop.is_set():
            t = time.perf_counter() - start
            if t >= self.duration:
                self.state.set_axes(self.norm_profile[-1][1])
                logger.info("[PROFILE] Completed")
                break

            # advance segment index
            while k + 1 < len(self.norm_profile) and self.norm_profile[k + 1][0] <= t:
                k += 1

            t0, p0 = self.norm_profile[k]
            t1, p1 = self.norm_profile[min(k + 1, len(self.norm_profile) - 1)]
            if t1 <= t0:
                alpha = 0.0
            else:
                alpha = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
            axes = [p0[i] + alpha * (p1[i] - p0[i]) for i in range(6)]
            try:
                self.state.set_axes(axes)
            except Exception as e:
                logger.error(f"[PROFILE] set_axes error: {e}")
            time.sleep(self.dt)

        # cleanup: clear player_thread reference if still pointing to us
        with self.state.lock:
            if self.state.player_thread is self:
                self.state.player_thread = None


class ODriveCANBridge(threading.Thread):
    """Bridge between RobotState and ODrive CAN bus using ODriveCanManager."""

    def __init__(self, state: RobotState, canbus="can0", axis_ids=None):
        super().__init__(daemon=True)
        self.state = state
        self.axis_ids = axis_ids or [0, 1, 2, 3, 4, 5]
        self.manager = None
        self._stop = threading.Event()

        #Apply the current state version to avoid auto applying the default by setting these to -1.  Perhaps reconsider this for desired auto init behavior later on
        self._applied_state_version = state.get_state_version()
        self._applied_home_version = state.get_home_version()
        self._applied_pret_version = state.get_pretension_version()

    def stop(self):
        self._stop.set()
        if self.manager:
            try:
                self.manager.close()
            except Exception:
                pass

    def run(self):
        logger.info("[ODRV] Starting ODriveCanManager…")
        try:
            # one bus, one listener
            self.manager = ODriveCanManager("can0")

            # register all axes + callbacks
            for aid in self.axis_ids:
                axis = self.manager.add_axis(aid)

                # encoder callback -> update RobotState
                axis.on_encoder(lambda pos, vel, i=aid:
                    self.state.set_axis_feedback(i, pos_estimate=pos, vel_estimate=vel))

                # bus voltage callback -> update RobotState
                axis.on_bus(lambda vbus, ibus, i=aid:
                    self.state.set_axis_feedback(i, bus_voltage=vbus, bus_current=ibus))

                # motor current callback
                axis.on_iq(lambda iq_set, iq_meas, i=aid:
                           self.state.set_axis_feedback(i, motor_current=iq_meas))

                # temperatures callback, if your ODriveCANSimple exposes it
                axis.on_temp(lambda fet, motor, i=aid:
                    self.state.set_axis_feedback(i, temp_fet=fet, temp_motor=motor))

                # heartbeat callback
                axis.on_heartbeat(lambda err, st, proc, i=aid:
                    self.state.set_axis_feedback(i, axis_error=err, axis_state=st, proc_result=proc))

                logger.info(f"[ODRV] axis {aid} registered callbacks")

            # main loop (~500 Hz)
            last_log = time.perf_counter()
            while not self._stop.is_set():
                st = self.state.get_state()
                sv = self.state.get_state_version()

                # Apply state transitions when version changes
                if sv != self._applied_state_version:
                    self._apply_state(st)
                    self._applied_state_version = sv

                # Apply HOME request (one-shot) when version changes
                hv = self.state.get_home_version()
                if hv != self._applied_home_version:
                    self._apply_home()
                    self._applied_home_version = hv

                # Apply PRETENSION request when version changes
                pv = self.state.get_pretension_version()
                if pv != self._applied_pret_version:
                    self._apply_pretension_mode()
                    self._applied_pret_version = pv

                # Stream setpoints if enabled
                if st == "enable":
                    pos_cmd_mm = self.state.get_pos_cmd()  # mm
                    pos_cmd_turns = mm_to_turns(pos_cmd_mm)  # turns for ODrive
                    for i, aid in enumerate(self.axis_ids):
                        try:
                            axis = self.manager.axes.get(aid)
                            if axis is not None:
                                axis.set_input_pos(pos_cmd_turns[i])
                        except Exception as e:
                            logger.warning(f"[ODRV] axis {aid} set_input_pos failed: {e}")
                elif st == "pretension":
                    upper_N, lower_N = self.state.get_pretension()

                    # Map upper/lower tension to per-axis torque commands
                    torque_cmd = [0.0] * 6
                    for i in (0, 2, 4):
                        torque_cmd[i] = upper_N * TORQUE_PER_TENSION
                    for i in (1, 3, 5):
                        torque_cmd[i] = lower_N * TORQUE_PER_TENSION

                    for i, aid in enumerate(self.axis_ids):
                        try:
                            axis = self.manager.axes.get(aid)
                            if axis is not None:
                                axis.set_input_torque(torque_cmd[i])
                        except Exception as e:
                            logger.warning(f"[ODRV] axis {aid} set_input_torque failed: {e}")

                # light heartbeat log
                now = time.perf_counter()
                if now - last_log >= 1.0:
                    logger.info(f"[ODRV] streaming {len(self.axis_ids)} axes, state={st}")
                    last_log = now

                time.sleep(0.002)  # ~500 Hz

        except Exception as e:
            logger.error(f"[ODRV] Bridge error: {e}")
        finally:
            if self.manager:
                try:
                    self.manager.close()
                except Exception:
                    pass
            logger.info("[ODRV] Manager stopped")

    def _apply_state(self, st: str):
        """Apply high-level state to all axes."""
        try:
            for aid in self.axis_ids:
                axis = self.manager.axes.get(aid) if self.manager else None
                if not axis:
                    continue

                if st == "enable":
                    axis.set_controller_mode(CONTROL_MODE_POSITION, INPUT_MODE_PASSTHROUGH)
                    logger.info(f"[ODRV] axis {aid}: POSITION + CLOSED_LOOP_CONTROL")
                    axis.set_axis_state(AxisState.CLOSED_LOOP_CONTROL)

                elif st == "pretension":
                    axis.set_controller_mode(CONTROL_MODE_TORQUE, INPUT_MODE_PASSTHROUGH)
                    logger.info(f"[ODRV] axis {aid}: TORQUE + CLOSED_LOOP_CONTROL")
                    axis.set_axis_state(AxisState.CLOSED_LOOP_CONTROL)

                elif st in ("disable", "estop"):
                    logger.info(f"[ODRV] axis {aid}: IDLE")
                    axis.set_axis_state(AxisState.IDLE)

        except Exception as e:
            logger.error(f"[ODRV] _apply_state error: {e}")

    def _apply_home(self):
        """
        HOME intent: do NOT move motors.
        We reset the estimator's absolute position (pos_estimate) to the GUI-provided
        home positions, and also align the streamed setpoints to those same values
        so the controller doesn't step.
        """
        if not self.manager:
            return

        home_pos_mm = self.state.get_home_pos()  # mm

        # 1) Update command setpoints first
        try:
            self.state.set_axes(home_pos_mm)  # keep cmd in mm
        except Exception as e:
            logger.error(f"[HOME] Failed to set_axes(home_pos_mm): {e}")

        # 2) Send Set_Absolute_Position to each axis (in turns)
        home_pos_turns = mm_to_turns(home_pos_mm)
        for i, aid in enumerate(self.axis_ids):
            axis = self.manager.axes.get(aid)
            if not axis:
                continue
            try:
                axis.set_absolute_position(home_pos_turns[i])
                axis.set_input_pos(home_pos_turns[i])
                logger.info(f"[HOME] axis {aid}: abs_pos <- {home_pos_mm[i]:.3f} mm ({home_pos_turns[i]:.4f} turns)")
            except Exception as e:
                logger.warning(f"[HOME] axis {aid} set_absolute_position failed: {e}")

    def _apply_pretension_mode(self):
        """Put all axes into torque control (passthrough) + closed loop."""
        try:
            for aid in self.axis_ids:
                axis = self.manager.axes.get(aid) if self.manager else None
                if not axis:
                    continue
                axis.set_controller_mode(CONTROL_MODE_TORQUE, INPUT_MODE_PASSTHROUGH)
                axis.set_axis_state(AxisState.CLOSED_LOOP_CONTROL)
            logger.info("[PRET] applied torque control mode to all axes")
        except Exception as e:
            logger.error(f"[PRET] _apply_pretension_mode error: {e}")


def udp_telemetry_sender(state: RobotState, udp_sock, stop_event):
    while not stop_event.is_set():
        try:
            controller_ip = state.get_controller_ip()
            if controller_ip:
                controller_addr = (controller_ip, UDP_TELEM_PORT)
                fb_pos_turns = state.get_pos_fbk()
                fb_vel_turnsps = state.get_vel_fbk()

                # Convert to mm + mm/s for the GUI
                fb_pos_mm = []
                fb_vel_mmps = []
                for i in range(6):
                    p = fb_pos_turns[i] if i < len(fb_pos_turns) else None
                    v = fb_vel_turnsps[i] if i < len(fb_vel_turnsps) else None
                    k = MM_PER_TURN[i]
                    fb_pos_mm.append(None if p is None else float(p) * k)
                    fb_vel_mmps.append(None if v is None else float(v) * k)

                bus_v = state.get_bus_voltage() or []
                bus_i = state.get_bus_current() or []
                motor_i = state.get_motor_current() or []
                temp_fet = state.get_temp_fet() or []
                temp_motor = state.get_temp_motor() or []
                axis_state = state.get_axis_state() or []
                axis_error = state.get_axis_error() or []
                msg = {
                    "t": time.time(),
                    "pos": fb_pos_mm,
                    "vel": fb_vel_mmps,
                    "bus_v": [None if v is None else float(v) for v in bus_v],
                    "bus_i": [None if i is None else float(i) for i in bus_i],
                    "motor_i": [None if x is None else float(x) for x in motor_i],
                    "temp_fet": [None if x is None else float(x) for x in temp_fet],
                    "temp_motor": [None if x is None else float(x) for x in temp_motor],
                    "axis_state": [None if x is None else int(x) for x in axis_state],
                    "axis_error": [None if x is None else int(x) for x in axis_error],
                }
                udp_sock.sendto(json.dumps(msg).encode("utf-8"), controller_addr)
        except Exception as e:
            logger.error(f"[UDP] Error sending telemetry: {e}")
        time.sleep(1.0 / TELEMETRY_RATE_HZ)



def axes_state_logger(state: RobotState):
    while True:
        try:
            pos = state.get_pos_fbk()
            vel = state.get_vel_fbk()
            bus = state.get_bus_voltage()
            busi = state.get_bus_current()
            temp_f = state.get_temp_fet()
            temp_m = state.get_temp_motor()
            st = state.get_state()

            fmt_pos = ", ".join("---" if x is None else f"{x:.3f}" for x in pos)
            fmt_vel = ", ".join("---" if v is None else f"{v:.3f}" for v in vel)
            fmt_bus = ", ".join("---" if b is None else f"{b:.2f}" for b in bus)
            fmt_busi = ", ".join("---" if i is None else f"{i:.2f}" for i in busi)
            fmt_tf = ", ".join("---" if x is None else f"{x:.1f}" for x in temp_f)
            fmt_tm = ", ".join("---" if x is None else f"{x:.1f}" for x in temp_m)

            logger.info(
                f"[LOG] State={st} "
                f"Pos=[{fmt_pos}] "
                f"Vel=[{fmt_vel}] "
                f"BusV=[{fmt_bus}]"
                f"BusI=[{fmt_busi}]"
                f"TempFET=[{fmt_tf}] "
                f"TempMotor=[{fmt_tm}]"
            )
        except Exception as e:
            logger.error(f"[LOG] Error: {e}")
        time.sleep(1.0)


def tcp_command_server(state: RobotState):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", TCP_CMD_PORT))
    except OSError as e:
        logger.error(f"[TCP] Bind failed: {e}")
        return
    srv.listen(1)
    logger.info(f"[TCP] Listening on :{TCP_CMD_PORT}")
    while True:
        conn, addr = srv.accept()
        state.set_controller_ip(addr[0])  # <-- save controller IP
        logger.info(f"[TCP] Controller connected from {addr}")
        state.set_controller_ip(addr[0])

        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        controller_addr = (addr[0], UDP_TELEM_PORT)
        state.start_telem(udp_sock, controller_addr)

        try:
            with conn, conn.makefile("r") as f:
                for line in f:
                    try:
                        msg = json.loads(line.strip())
                        mtype = msg.get("type")
                        if mtype == "axes":
                            pos_mm = _coerce_vec6_to_mm(msg, "positions")
                            state.set_axes(pos_mm)
                        elif mtype == "state":
                            state.set_state(msg.get("value", "disable"))
                        elif mtype == "pretension":
                            upper = float(msg.get("upper_N", 0.0))
                            lower = float(msg.get("lower_N", 0.0))
                            state.request_pretension(upper, lower)
                        elif mtype == "home":
                            home_mm = _coerce_vec6_to_mm(msg, "home_pos")
                            state.request_home(home_mm)
                        elif mtype == "profile_upload":
                            profile_mm = msg.get("profile", [])

                            # Convert each row: [t, a1..a6] where a1..a6 are mm -> turns
                            profile_turns = []
                            for row in profile_mm:
                                if not isinstance(row, (list, tuple)) or len(row) < 7:
                                    raise ValueError("each profile row must be [t, a1..a6]")
                                t = float(row[0])
                                axes_mm = [float(x) for x in row[1:7]]
                                axes_turns = _mm_to_turns(axes_mm)
                                profile_turns.append([t] + axes_turns)

                            state.set_profile(profile_turns)
                        elif mtype == "profile_start":
                            rate_hz = float(msg.get("rate_hz", 100.0))
                            state.start_profile(rate_hz)
                        elif mtype == "profile_stop":
                            state.stop_profile()
                        else:
                            logger.warning(f"[TCP] Unknown command: {mtype}")
                    except Exception as e:
                        logger.error(f"[TCP] Bad command: {e}")
        except Exception as e:
            logger.error(f"[TCP] Connection error: {e}")
        finally:
            state.stop_profile()
            state.stop_telem()
            logger.info("[TCP] Controller disconnected")


if __name__ == "__main__":
    state = RobotState()
    can_ok = ensure_can_interface_up(ODRIVE_INTERFACE, ODRIVE_BITRATE)
    if not can_ok:
        logger.warning("[CAN] Continuing without CAN up")

    odrv_bridge = ODriveCANBridge(state)
    odrv_bridge.start()

    threading.Thread(target=tcp_command_server, args=(state,), daemon=True).start()
    threading.Thread(target=axes_state_logger, args=(state,), daemon=True).start()

    logger.info("Robot server running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        odrv_bridge.stop()
