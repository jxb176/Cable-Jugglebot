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
        # 6-axis targets (turns) and robot state
        self.axes = [0.0] * 6          # positions in turns
        self.state = "disable"         # enable|disable|estop
        # Profile storage and player
        self.profile = []
        self.player_thread = None
        # Feedback storage (pos/vel estimates), to be updated by your drive bridge
        self._fb_pos = [0.0] * 6
        self._fb_vel = [0.0] * 6

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

    # Feedback setters/getters
    def update_feedback(self, pos=None, vel=None):
        """Update measured feedback (pos/vel arrays length 6)."""
        with self.lock:
            if pos is not None and len(pos) == 6:
                self._fb_pos = [float(x) for x in pos]
            if vel is not None and len(vel) == 6:
                self._fb_vel = [float(x) for x in vel]

    def get_feedback(self):
        with self.lock:
            return (list(self._fb_pos), list(self._fb_vel))

def udp_telemetry_sender(state: RobotState):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        time.sleep(0.1)
        ctrl_ip = state.get_controller_ip()
        if not ctrl_ip:
            continue
        # Prefer real feedback if available; otherwise fall back to targets + zeros
        fb_pos, fb_vel = state.get_feedback()
        if not any(fb_pos):  # simple heuristic; customize as needed
            fb_pos = state.get_axes()
        msg = {"t": time.time(), "pos": [float(x) for x in fb_pos], "vel": [float(x) for x in fb_vel]}
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


# Fallback: define a no-op ODriveCANBridge if the real one isn't present
try:
    ODriveCANBridge  # type: ignore[name-defined]
except NameError:
    class ODriveCANBridge(threading.Thread):
        """No-op ODrive bridge so the server can run without drive I/O."""
        def __init__(self, state):
            super().__init__(daemon=True)
            self._stop = threading.Event()
        def stop(self):
            self._stop.set()
        def run(self):
            try:
                logger.info("ODriveCANBridge stub active (no CAN I/O)")
            except Exception:
                print("ODriveCANBridge stub active (no CAN I/O)")
            while not self._stop.is_set():
                time.sleep(0.5)

# Fallback: define tcp_command_server if missing
try:
    tcp_command_server  # type: ignore[name-defined]
except NameError:
    def tcp_command_server(state):
        """Minimal TCP command server accepting axes/state/profile messages."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("0.0.0.0", TCP_CMD_PORT))
        except OSError as e:
            try:
                logger.error(f"[TCP] Bind failed on :{TCP_CMD_PORT} ({e}). Exiting tcp thread.")
            except Exception:
                print(f"[TCP] Bind failed on :{TCP_CMD_PORT} ({e}). Exiting tcp thread.")
            return
        srv.listen(1)
        try:
            logger.info(f"[TCP] Listening on :{TCP_CMD_PORT}")
        except Exception:
            print(f"[TCP] Listening on :{TCP_CMD_PORT}")
        while True:
            conn, addr = srv.accept()
            try:
                logger.info(f"[TCP] Controller connected from {addr}")
            except Exception:
                print(f"[TCP] Controller connected from {addr}")
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
                                state.set_profile(msg.get("profile", []))
                            elif mtype == "profile_start":
                                rate_hz = float(msg.get("rate_hz", 100.0))
                                state.start_profile(rate_hz)
                            elif mtype == "profile_stop":
                                state.stop_profile()
                            else:
                                try:
                                    logger.warning(f"[TCP] Unknown command type: {mtype}")
                                except Exception:
                                    print(f"[TCP] Unknown command type: {mtype}")
                        except Exception as e:
                            try:
                                logger.error(f"[TCP] Bad command: {e}")
                            except Exception:
                                print(f"[TCP] Bad command: {e}")
            except Exception as e:
                try:
                    logger.error(f"[TCP] Connection error: {e}")
                except Exception:
                    print(f"[TCP] Connection error: {e}")
            finally:
                state.stop_profile()
                try:
                    logger.info("[TCP] Controller disconnected")
                except Exception:
                    print("[TCP] Controller disconnected")

# Fallback: define udp_telemetry_sender if missing
try:
    udp_telemetry_sender  # type: ignore[name-defined]
except NameError:
    def udp_telemetry_sender(state):
        """Sends pos/vel arrays (or fallbacks) over UDP at ~10 Hz."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while True:
            time.sleep(0.1)
            ctrl_ip = state.get_controller_ip()
            if not ctrl_ip:
                continue
            # Try to use feedback if available; fall back to targets & zeros
            try:
                fb_pos, fb_vel = state.get_feedback()
            except Exception:
                fb_pos, fb_vel = ([], [])
            if not fb_pos or len(fb_pos) != 6:
                fb_pos = state.get_axes()
            if not fb_vel or len(fb_vel) != 6:
                fb_vel = [0.0] * 6
            msg = {
                "t": time.time(),
                "pos": [float(x) for x in fb_pos],
                "vel": [float(x) for x in fb_vel],
            }
            try:
                sock.sendto(json.dumps(msg).encode("utf-8"), (ctrl_ip, UDP_TELEM_PORT))
            except Exception as e:
                try:
                    logger.error(f"[UDP] Telemetry send error: {e}")
                except Exception:
                    print(f"[UDP] Telemetry send error: {e}")

# Fallback: define axes_state_logger if missing
try:
    axes_state_logger  # type: ignore[name-defined]
except NameError:
    def axes_state_logger(state):
        """Logs current state and targets at 1 Hz."""
        while True:
            try:
                axes = state.get_axes()
                st = state.get_state()
                logger.info(f"[LOG] State={st} Axes(turns)=[" +
                            ", ".join(f"{x:.3f}" for x in axes) + "]")
            except Exception as e:
                try:
                    logger.error(f"[LOG] Error reading state/axes: {e}")
                except Exception:
                    print(f"[LOG] Error reading state/axes: {e}")
            time.sleep(1.0)

# --- main ---
if __name__ == "__main__":
    state = RobotState()

    # Start ODrive CAN bridge (real or stub)
    odrv_bridge = ODriveCANBridge(state)
    odrv_bridge.start()

    threading.Thread(target=tcp_command_server, args=(state,), daemon=True).start()
    threading.Thread(target=udp_telemetry_sender, args=(state,), daemon=True).start()
    threading.Thread(target=axes_state_logger, args=(state,), daemon=True).start()
    try:
        logger.info("Robot server running. Press Ctrl+C to exit.")
    except Exception:
        print("Robot server running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        try:
            logger.info("Shutting down...")
        except Exception:
            print("Shutting down...")
        odrv_bridge.stop()