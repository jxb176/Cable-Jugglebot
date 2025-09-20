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
from typing import List, Tuple, Optional

TCP_CMD_PORT = 5555
UDP_TELEM_PORT = 5556

# -------- ODrive CAN configuration --------
ODRIVE_INTERFACE = "can0"            # e.g., "can0" or "vcan0"
ODRIVE_BITRATE = 1_000_000           # 1 Mbps
AXIS_NODE_IDS = [0, 1, 2, 3, 4, 5]
ODRIVE_COMMAND_RATE_HZ = 200.0
ODRIVE_LOG_RATE_HZ = 2.0
TELEMETRY_RATE_HZ = 50.0
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
        # 6-axis commanded position targets (turns)
        self.axes_pos_cmd = [0.0] * 6
        self.state = "disable"         # enable|disable|estop
        self.state_version = 0         # bump on every set_state() to notify bridge
        self.profile = []
        self.player_thread = None
        # Per-axis measured feedback (pos_estimate / vel_estimate)
        self.axes_pos_estimate = [None] * 6
        self.axes_vel_estimate = [None] * 6

        # Telemetry management
        self.telem_thread = None
        self.telem_stop = threading.Event()

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
            self.axes_pos_cmd = [float(x) for x in positions]
        logger.info("Axes target set: " + ", ".join(f"{x:.4f}" for x in self.axes_pos_cmd))

    def get_pos_cmd(self):
        with self.lock:
            return list(self.axes_pos_cmd)

    def get_pos_fbk(self):
        with self.lock:
            return list(self.axes_pos_estimate)

    def get_vel_fbk(self):
        with self.lock:
            return list(self.axes_vel_estimate)

    def set_state(self, value: str):
        value = str(value).lower()
        if value not in ("enable", "disable", "estop"):
            raise ValueError("state must be one of: enable, disable, estop")
        with self.lock:
            self.state = value
            self.state_version += 1
        logger.info(f"State set to: {value} (version {self.state_version})")

    def get_state(self) -> str:
        with self.lock:
            return self.state

    def get_state_version(self) -> int:
        with self.lock:
            return self.state_version

    # Feedback setters/getters (per-axis)
    def set_axis_feedback(self, axis_id: int, pos_estimate=None, vel_estimate=None):
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

    def get_feedback(self):
        with self.lock:
            return (list(self.axes_pos_estimate), list(self.axes_vel_estimate))

    # --- Telemetry lifecycle ---
    def start_telem(self, udp_sock, controller_addr):
        self.stop_telem()
        self.telem_stop.clear()
        t = threading.Thread(
            target=udp_telemetry_sender,
            args=(self, udp_sock, controller_addr, self.telem_stop),
            daemon=True
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

    # --- Profile management (unchanged) ---
    # ... keep your ProfilePlayer methods here ...


def udp_telemetry_sender(state: RobotState, udp_sock, controller_addr, stop_event):
    """Send telemetry (pos/vel) to controller until stop_event is set."""
    while not stop_event.is_set():
        try:
            fb_pos = state.get_pos_fbk()
            fb_vel = state.get_vel_fbk()
            msg = {
                "t": time.time(),
                "pos": [None if v is None else float(v) for v in fb_pos],
                "vel": [None if v is None else float(v) for v in fb_vel],
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
            logger.error(f"[LOG] Error reading state/axes: {e}")
        time.sleep(1.0)


def tcp_command_server(state: RobotState):
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

        # Start telemetry thread for this controller
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
                            logger.warning(f"[TCP] Unknown command type: {mtype}")
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

    # Ensure CAN interface is up before starting ODrive bridge
    can_ok = ensure_can_interface_up(ODRIVE_INTERFACE, ODRIVE_BITRATE)
    if not can_ok:
        logger.warning(f"[CAN] Continuing without {ODRIVE_INTERFACE} being UP (ODrive bridge may run in simulation or fail)")

    # Start ODrive CAN bridge (async driver + feedback logging)
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
