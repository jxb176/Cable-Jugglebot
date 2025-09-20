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
        self.profile = []
        self.player_thread = None
        self.axes_pos_estimate = [None] * 6
        self.axes_vel_estimate = [None] * 6
        self.axes_bus_voltage = [None] * 6
        self.telem_thread = None
        self.telem_stop = threading.Event()

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

    def set_axis_feedback(self, axis_id: int, pos_estimate=None, vel_estimate=None, bus_voltage=None):
        """Store measured feedback for a single axis index (0..5)."""
        if not (0 <= int(axis_id) < 6):
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

    def get_bus_voltage(self):
        """Return list of bus voltage values (may contain None)."""
        with self.lock:
            return list(self.axes_bus_voltage)

    def set_axes(self, positions):
        if not isinstance(positions, (list, tuple)) or len(positions) != 6:
            raise ValueError("positions must be length-6 list/tuple")
        with self.lock:
            self.axes_pos_cmd = [float(x) for x in positions]
        logger.info("Axes target set: " + ", ".join(f"{x:.4f}" for x in self.axes_pos_cmd))

    def set_state(self, value: str):
        value = str(value).lower()
        if value not in ("enable", "disable", "estop"):
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
    """Streams axes targets to ODrive over CAN (async) and logs feedback callbacks."""

    def __init__(self, state: RobotState):
        super().__init__(daemon=True)
        self.state = state
        self._stop = threading.Event()
        self._drivers: list[tuple[int, any]] = []
        self._applied_state_version = -1

    def stop(self):
        self._stop.set()

    async def _start_all(self):
        """Initialize all ODriveCAN drivers and attach feedback callbacks."""
        if odc is None:
            logger.warning("[ODRV] odrive_can module missing; running in simulation mode")
            return

        self._drivers = []
        for axis_id in AXIS_NODE_IDS:
            try:
                drv = odc.ODriveCAN(axis_id=axis_id)

                # Proper callback signature: (msg: CanMsg, caller: ODriveCAN)
                def make_feedback_cb(aid):
                    def cb(msg, caller=None):
                        #logger.debug(f"[ODRV] raw feedback axis {aid}: {msg}")
                        self._on_feedback(aid, msg)
                    return cb

                drv.feedback_callback = make_feedback_cb(axis_id)
                try:
                    logger.info(f"[ODRV] axis {axis_id}: starting driver")
                    await asyncio.wait_for(drv.start(), timeout=1.0)
                    self._drivers.append((axis_id, drv))
                    logger.info(f"[ODRV] axis {axis_id}: driver started")
                except asyncio.TimeoutError:
                    logger.warning(f"[ODRV] axis {axis_id}: drv.start() timed out")
                except Exception as e:
                    logger.warning(f"[ODRV] axis {axis_id} driver init failed: {e}")

            except Exception as e:
                logger.warning(f"[ODRV] axis {axis_id} driver init failed: {e}")

        if not self._drivers:
            logger.warning("[ODRV] No drivers successfully initialized")
        else:
            logger.info(f"[ODRV] drivers initialized: {[aid for aid, _ in self._drivers]}")

    def _on_feedback(self, axis_id: int, msg):
        """
        Handle feedback from a single axis.

        msg is a CanMsg object from odrive_can.
        """
        try:
            # Try to decode useful fields
            pos_val = None
            vel_val = None
            bus_v = None

            # Many CanMsg objects have a .data dict with decoded signals
            if hasattr(msg, "data") and isinstance(msg.data, dict):
                pos_val = msg.data.get("Pos_Estimate")
                vel_val = msg.data.get("Vel_Estimate")
                bus_v = msg.data.get("Bus_Voltage")

            # Some versions may use .signals instead
            if pos_val is None and hasattr(msg, "signals"):
                pos_val = msg.signals.get("Pos_Estimate")
            if vel_val is None and hasattr(msg, "signals"):
                vel_val = msg.signals.get("Vel_Estimate")
            if bus_v is None and hasattr(msg, "signals"):
                bus_v = msg.signals.get("Bus_Voltage")

            #logger.debug(f"[ODRV] axis {axis_id} decoded pos={pos_val}, vel={vel_val}")

            if (pos_val is not None) or (vel_val is not None) or (bus_v is not None):
                self.state.set_axis_feedback(
                    axis_id,
                    pos_estimate = pos_val,
                    vel_estimate = vel_val,
                    bus_voltage = bus_v,
                )

        except Exception as e:
            logger.exception(f"[ODRV] Exception in _on_feedback for axis {axis_id}: {e}")

    async def _apply_state(self, st: str):
        """Apply high-level state (enable/disable/estop) to all axes."""
        if not self._drivers:
            return

        try:
            if st == "enable":
                for idx, (aid, drv) in enumerate(self._drivers):
                    logger.info(f"[ODRV] axis {aid}: enabling CLOSED_LOOP_CONTROL")
                    try:
                        await drv.set_axis_state("CLOSED_LOOP_CONTROL")
                        cmd = self.state.get_pos_cmd()
                        setp = float(cmd[idx]) if idx < len(cmd) else 0.0
                        drv.set_input_pos(setp)
                    except Exception as e:
                        logger.warning(f"[ODRV] axis {aid} enable failed: {e}")

            elif st in ("disable", "estop"):
                for aid, drv in self._drivers:
                    logger.info(f"[ODRV] axis {aid}: setting IDLE")
                    try:
                        await drv.set_axis_state("IDLE")
                    except Exception as e:
                        logger.warning(f"[ODRV] axis {aid} disable failed: {e}")
        except Exception as e:
            logger.error(f"[ODRV] _apply_state('{st}') failed: {e}")

    async def _stream_positions(self):
        """Main loop: stream commands + log feedback."""
        dt_cmd = 1.0 / max(1e-3, ODRIVE_COMMAND_RATE_HZ)
        dt_log = 1.0 / max(1e-3, ODRIVE_LOG_RATE_HZ)
        last_cmd = time.perf_counter()
        last_log = time.perf_counter()

        while not self._stop.is_set():
            now = time.perf_counter()
            st = self.state.get_state()
            sv = self.state.get_state_version()

            # Handle state change
            if sv != self._applied_state_version:
                await self._apply_state(st)
                self._applied_state_version = sv

            # Send setpoints if enabled
            if self._drivers and st == "enable" and (now - last_cmd) >= dt_cmd:
                positions = self.state.get_pos_cmd()
                for i, (aid, drv) in enumerate(self._drivers):
                    try:
                        drv.set_input_pos(float(positions[i]))
                    except Exception as e:
                        logger.error(f"[ODRV] axis {aid} set_input_pos failed: {e}")
                last_cmd = now

            # Periodic log
            if (now - last_log) >= dt_log:
                logger.info(f"[ODRV] streaming {len(self._drivers)} axes, state={st}")
                last_log = now

            await asyncio.sleep(0.002)

    async def _run_async(self):
        """Async entry point for the bridge."""
        logger.info("[ODRV] _run_async starting _start_all")
        await self._start_all()
        logger.info("[ODRV] _run_async finished _start_all, entering _stream_positions")
        await self._stream_positions()

    def run(self):
        """Thread entry point — runs an asyncio loop."""
        logger.info("[ODRV] ODriveCANBridge thread started, entering asyncio.run")
        try:
            asyncio.run(self._run_async())
        except Exception as e:
            logger.error(f"[ODRV] asyncio loop error: {e}")



def udp_telemetry_sender(state: RobotState, udp_sock, stop_event):
    while not stop_event.is_set():
        try:
            controller_ip = state.get_controller_ip()
            if controller_ip:
                controller_addr = (controller_ip, UDP_TELEM_PORT)
                fb_pos = state.get_pos_fbk()
                fb_vel = state.get_vel_fbk()
                bus_v = state.get_bus_voltage()
                msg = {
                    "t": time.time(),
                    "pos": [None if v is None else float(v) for v in fb_pos],
                    "vel": [None if v is None else float(v) for v in fb_vel],
                    "bus_v": [None if v is None else float(v) for v in bus_v],
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
            st = state.get_state()
            fmt_pos = ", ".join("---" if x is None else f"{x:.3f}" for x in pos)
            fmt_vel = ", ".join("---" if v is None else f"{v:.3f}" for v in vel)
            logger.info(f"[LOG] State={st} Pos=[{fmt_pos}] Vel=[{fmt_vel}]")
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
                            state.set_axes(msg.get("positions", []))
                        elif mtype == "state":
                            state.set_state(msg.get("value", "disable"))
                        elif mtype == "profile_upload":
                            profile = msg.get("profile", [])
                            state.set_profile(profile)
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
