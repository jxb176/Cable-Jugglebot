# robot_server.py
import socket
import json
import time
import threading
import random
import os
import logging
from datetime import datetime
import subprocess
import asyncio  # <-- make asyncio available at module scope

TCP_CMD_PORT = 5555
UDP_TELEM_PORT = 5556

# -------- ODrive CAN configuration --------
ODRIVE_INTERFACE = "can0"            # e.g., "can0" or "vcan0"
ODRIVE_BITRATE = 1_000_000           # 1 Mbps
#AXIS_NODE_IDS = [0, 1, 2, 3, 4, 5]
AXIS_NODE_IDS = [0]
ODRIVE_COMMAND_RATE_HZ = 200.0
ODRIVE_LOG_RATE_HZ = 2.0
# Ensure env var for libraries that require CAN_CHANNEL
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
    logger.setLevel(logging.INFO)
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
    """Ensure CAN interface is up with a given bitrate. Returns True if up."""
    try:
        # Check current status
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
            logger.warning(f"[CAN] '{ifname}' not found or not available: {res.stderr.strip()}")

        logger.info(f"[CAN] Bringing up {ifname} @ {bitrate} bps")
        # Bring down (ignore errors)
        subprocess.run(["ip", "link", "set", ifname, "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Configure type/bitrate
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
        # Bring up
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

        # Verify
        ver = subprocess.run(
            ["ip", "link", "show", ifname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if ver.returncode == 0 and (" state up " in ver.stdout.lower() or "<up," in ver.stdout.lower() or "up>" in ver.stdout.lower()):
            logger.info(f"[CAN] {ifname} is UP")
            return True
        logger.error(f"[CAN] {ifname} did not come UP; status: {ver.stdout.strip()}")
        return False
    except FileNotFoundError:
        logger.error("[CAN] 'ip' command not found. Install iproute2 or run on a system with 'ip'.")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"[CAN] Timeout while configuring {ifname}")
        return False
    except Exception as e:
        logger.error(f"[CAN] Unexpected error: {e}")
        return False


class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.controller_ip = None
        self.axes = [0.0] * 6          # turns
        self.state = "disable"         # enable|disable|estop
        self.profile = []
        self.player_thread = None

    def set_controller_ip(self, ip):
        with self.lock:
            self.controller_ip = ip
        logger.info(f"Controller IP set to {ip}")

    def get_controller_ip(self):
        with self.lock:
            return self.controller_ip

    def set_axes(self, positions):
        if not isinstance(positions, (list, tuple)) or len(positions) != 6:
            raise ValueError("positions must be length-6 list/tuple")
        with self.lock:
            self.axes = [float(x) for x in positions]
        logger.info("Axes target set: " + ", ".join(f"{x:.4f}" for x in self.axes))

    def get_axes(self):
        with self.lock:
            return list(self.axes)

    def set_state(self, value: str):
        value = str(value).lower()
        if value not in ("enable", "disable", "estop"):
            raise ValueError("state must be one of: enable, disable, estop")
        with self.lock:
            self.state = value
        logger.info(f"State set to: {value}")

    def get_state(self) -> str:
        with self.lock:
            return self.state

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
        logger.info(f"Profile uploaded: {len(prof)} points, duration {prof[-1][0]-prof[0][0]:.3f}s")

    def get_profile(self):
        with self.lock:
            return list(self.profile)

    def start_profile(self, rate_hz: float):
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
        with self.lock:
            player = self.player_thread
            self.player_thread = None
        if player and player.is_alive():
            player.stop()
            player.join(timeout=1.0)
            logger.info("Profile stopped")


class ProfilePlayer(threading.Thread):
    """Plays a time-position profile with linear interpolation at fixed rate."""

    def stop(self):
        self._stop.set()

    def run(self):
        logger.info(f"[PROFILE] Starting playback at {1.0/self.dt:.1f} Hz, duration {self.duration:.3f}s")
        start = time.perf_counter()
        k = 0
        while not self._stop.is_set():
            t = time.perf_counter() - start
            if t >= self.duration:
                self.state.set_axes(self.norm_profile[-1][1])
                logger.info("[PROFILE] Completed")
                break
            while k + 1 < len(self.norm_profile) and self.norm_profile[k + 1][0] < t:
                k += 1
            t0, p0 = self.norm_profile[k]
            t1, p1 = self.norm_profile[min(k + 1, len(self.norm_profile) - 1)]
            if t1 <= t0:
                alpha = 0.0
            else:
                alpha = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
            axes = [p0[i] + alpha * (p1[i] - p0[i]) for i in range(6)]
            self.state.set_axes(axes)
            time.sleep(self.dt)
        with self.state.lock:
            if self.state.player_thread is self:
                self.state.player_thread = None


class ODriveCANBridge(threading.Thread):
    """Streams axes targets to ODrive over CAN (async) and logs feedback callbacks."""
    def __init__(self, state: RobotState):
        super().__init__(daemon=True)
        self.state = state
        self._stop = threading.Event()
        self._drivers = []              # list[(axis_id, drv)]
        self._last_state = None
        self._last_log = 0.0
        self._feedback_seen = {aid: 0 for aid in AXIS_NODE_IDS}

    def stop(self):
        self._stop.set()

    def _feedback_cb(self, axis_id: int):
        def _cb(msg, caller):
            print(f"[ODRV] axis={axis_id} feedback: {msg}")
            self._feedback_seen[axis_id] = self._feedback_seen.get(axis_id, 0) + 1
        return _cb

    async def _start_all(self):
        """Create and start ODriveCAN instances for all node IDs."""
        if odc is None:
            logger.warning("odrive_can not available; running in simulation mode (no CAN I/O)")
            return

        # Try to create a python-can Bus to pass into ODriveCAN
        can_bus = None
        try:
            import can  # python-can
            # Use 'interface' kw (bustype is deprecated in recent python-can)
            can_bus = can.interface.Bus(interface="socketcan", channel=ODRIVE_INTERFACE)
            logger.info(f"[CAN] python-can bus created for {ODRIVE_INTERFACE}")
        except Exception as e:
            logger.warning(f"[CAN] Could not create python-can bus for {ODRIVE_INTERFACE}: {e}")

        for aid in AXIS_NODE_IDS:
            try:
                drv = None
                last_err = None

                ctor_attempts = []
                if can_bus is not None:
                    # Prefer passing an actual bus object
                    ctor_attempts.append(("kw axis_id+busObj", lambda: odc.ODriveCAN(axis_id=aid, bus=can_bus)))
                # Fallbacks that rely on env var CAN_CHANNEL
                ctor_attempts.extend([
                    ("kw axis_id only (env CAN_CHANNEL)", lambda: odc.ODriveCAN(axis_id=aid)),
                ])

                for label, factory in ctor_attempts:
                    try:
                        drv = factory()
                        logger.info(f"ODrive axis {aid}: constructed with '{label}'")
                        # Sanity check: avoid keeping a driver if its internal bus is a str
                        if hasattr(drv, "_bus") and isinstance(getattr(drv, "_bus"), str):
                            raise TypeError("Driver _bus is str; expected bus object")
                        break
                    except Exception as e:
                        last_err = e
                        drv = None
                        continue

                if drv is None:
                    raise RuntimeError(f"ODriveCAN constructor not compatible for axis {aid}: {last_err}")

                drv.feedback_callback = self._feedback_cb(aid)
                await drv.start()

                # Optional: configure controller mode for position control (per example)
                drv.check_errors()
                drv.set_controller_mode("POSITION_CONTROL", "POS_FILTER")
                drv.set_linear_count(0)

                self._drivers.append((aid, drv))
                logger.info(f"ODrive axis {aid} started on '{ODRIVE_INTERFACE}'")
            except Exception as e:
                logger.error(f"Failed to init ODrive axis {aid}: {e}")

    async def _apply_state(self, st: str):
        if not self._drivers:
            return
        try:
            if st == "enable":
                for _, drv in self._drivers:
                    await drv.set_axis_state("CLOSED_LOOP_CONTROL")
            elif st == "disable":
                for _, drv in self._drivers:
                    await drv.set_axis_state("IDLE")
            elif st == "estop":
                for _, drv in self._drivers:
                    try:
                        await drv.set_axis_state("IDLE")
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Failed applying state '{st}' to ODrive: {e}")

    async def _stream_positions(self):
        dt_cmd = 1.0 / max(1e-3, ODRIVE_COMMAND_RATE_HZ)
        dt_log = 1.0 / max(1e-3, ODRIVE_LOG_RATE_HZ)
        t0 = time.perf_counter()
        last_cmd = t0
        last_log = t0

        while not self._stop.is_set():
            now = time.perf_counter()
            st = self.state.get_state()

            if st != self._last_state:
                await self._apply_state(st)
                self._last_state = st

            if self._drivers and st == "enable" and (now - last_cmd) >= dt_cmd:
                positions = self.state.get_axes()
                for i, (aid, drv) in enumerate(self._drivers):
                    try:
                        drv.set_input_pos(float(positions[i]))
                    except Exception as e:
                        logger.error(f"Axis {aid} set_input_pos failed: {e}")
                last_cmd = now

            if (now - last_log) >= dt_log:
                summary = ", ".join(f"{aid}:{self._feedback_seen.get(aid,0)}"
                                    for aid, _ in self._drivers)
                logger.info(f"[ODRV] feedback counts: {summary if summary else 'no drivers'}")
                last_log = now

            await asyncio.sleep(0.002)

    async def _run_async(self):
        await self._start_all()
        await self._apply_state(self.state.get_state())
        await self._stream_positions()

    def run(self):
        if odc is None:
            logger.warning("ODrive bridge running in simulation mode (odrive_can missing)")
            while not self._stop.is_set():
                time.sleep(0.5)
            return
        try:
            asyncio.run(self._run_async())
        except Exception as e:
            logger.error(f"ODrive asyncio loop error: {e}")


def tcp_command_server(state: RobotState):
    """Accepts a single TCP client, receives newline-delimited JSON commands."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", TCP_CMD_PORT))
    except OSError as e:
        logger.error(f"[TCP] Bind failed on :{TCP_CMD_PORT} ({e}). Another instance running? Server thread exiting.")
        return
    srv.listen(1)
    logger.info(f"[TCP] Listening on :{TCP_CMD_PORT}")
    while True:
        conn, addr = srv.accept()
        logger.info(f"[TCP] Controller connected from {addr}")
        state.set_controller_ip(addr[0])
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
                            logger.warning(f"[TCP] Unknown command type: {mtype}")
                    except Exception as e:
                        logger.error(f"[TCP] Bad command: {e}")
        except Exception as e:
            logger.error(f"[TCP] Connection error: {e}")
        finally:
            state.stop_profile()
            logger.info("[TCP] Controller disconnected")


def udp_telemetry_sender(state: RobotState):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        time.sleep(0.1)
        ctrl_ip = state.get_controller_ip()
        if not ctrl_ip:
            continue
        axes = state.get_axes()
        a1 = axes[0] if axes else 0.0
        val = a1 + random.uniform(-0.05, 0.05)
        msg = {"t": time.time(), "val": float(val)}
        try:
            sock.sendto(json.dumps(msg).encode("utf-8"), (ctrl_ip, UDP_TELEM_PORT))
        except Exception as e:
            logger.error(f"[UDP] Telemetry send error: {e}")


def axes_state_logger(state: RobotState):
    while True:
        try:
            axes = state.get_axes()
            st = state.get_state()
            logger.info(f"[LOG] State={st} Axes(turns)=[" +
                        ", ".join(f"{x:.3f}" for x in axes) + "]")
        except Exception as e:
            logger.error(f"[LOG] Error reading state/axes: {e}")
        time.sleep(1.0)


if __name__ == "__main__":
    state = RobotState()

    # Ensure CAN interface is up before starting ODrive bridge
    can_ok = ensure_can_interface_up(ODRIVE_INTERFACE, ODRIVE_BITRATE)
    if not can_ok:
        logger.warning(f"[CAN] Continuing without {ODRIVE_INTERFACE} being UP (ODrive bridge may run in simulation or fail)")

    # Start ODrive CAN bridge (async driver + feedback logging)
    odrv_bridge = ODriveCANBridge(state)
    odrv_bridge.start()

    threading.Thread(target=tcp_command_server, args=(state,), daemon=True).start()
    threading.Thread(target=udp_telemetry_sender, args=(state,), daemon=True).start()
    threading.Thread(target=axes_state_logger, args=(state,), daemon=True).start()
    logger.info("Robot server running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        odrv_bridge.stop()