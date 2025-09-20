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
ODRIVE_COMMAND_RATE_HZ = 200.0
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
        self.telem_thread = None
        self.telem_stop = threading.Event()

    def set_controller_ip(self, ip):
        with self.lock:
            self.controller_ip = ip
        logger.info(f"Controller IP set to {ip}")

    def get_pos_cmd(self):
        with self.lock:
            return list(self.axes_pos_cmd)

    def get_pos_fbk(self):
        with self.lock:
            return list(self.axes_pos_estimate)

    def get_vel_fbk(self):
        with self.lock:
            return list(self.axes_vel_estimate)

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

    # --- Telemetry lifecycle ---
    def start_telem(self, udp_sock, controller_addr):
        self.stop_telem()
        self.telem_stop.clear()
        t = threading.Thread(
            target=udp_telemetry_sender,
            args=(self, udp_sock, controller_addr, self.telem_stop),
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


class ODriveCANBridge(threading.Thread):
    def __init__(self, state: RobotState):
        super().__init__(daemon=True)
        self.state = state
        self._stop = threading.Event()
        self._drivers = []
        self._applied_state_version = -1

    def stop(self):
        self._stop.set()

    async def _start_all(self):
        if odc is None:
            logger.warning("[ODRV] odrive_can missing; sim mode")
            return
        self._drivers = []
        for axis_id in AXIS_NODE_IDS:
            try:
                drv = odc.ODriveCAN(axis_id=axis_id, interface=ODRIVE_INTERFACE)

                def make_cb(aid):
                    def cb(fb):
                        self._on_feedback(aid, fb)
                    return cb

                drv.feedback_callback = make_cb(axis_id)
                await drv.start()

                drv.check_errors()

                # Optional: configure controller mode
                drv.set_controller_mode("POSITION_CONTROL", "POS_FILTER")

                # Reset encoder count
                drv.set_linear_count(0)

                # Put into closed-loop so it streams feedback
                await drv.set_axis_state("IDLE")

                self._drivers.append((axis_id, drv))
                logger.info(f"[ODRV] axis {axis_id} started")
            except Exception as e:
                logger.warning(f"[ODRV] axis {axis_id} init failed: {e}")

    def _on_feedback(self, axis_id: int, fb):
        try:
            pos_val = fb.get("Pos_Estimate") if isinstance(fb, dict) else None
            vel_val = fb.get("Vel_Estimate") if isinstance(fb, dict) else None
            try:
                pos_val = float(pos_val) if pos_val is not None else None
            except Exception:
                pos_val = None
            try:
                vel_val = float(vel_val) if vel_val is not None else None
            except Exception:
                vel_val = None
            if pos_val is not None or vel_val is not None:
                self.state.set_axis_feedback(axis_id, pos_val, vel_val)
        except Exception as e:
            logger.error(f"[ODRV] feedback error: {e}")

    async def _apply_state(self, st: str):
        if not self._drivers:
            return
        try:
            if st == "enable":
                for idx, (aid, drv) in enumerate(self._drivers):
                    await drv.set_axis_state("CLOSED_LOOP_CONTROL")
                    cmd = self.state.get_pos_cmd()
                    setp = float(cmd[idx]) if idx < len(cmd) else 0.0
                    drv.set_input_pos(setp)
            elif st in ("disable", "estop"):
                for aid, drv in self._drivers:
                    await drv.set_axis_state("IDLE")
        except Exception as e:
            logger.error(f"[ODRV] apply_state error: {e}")

    async def _stream_positions(self):
        dt_cmd = 1.0 / max(1e-3, ODRIVE_COMMAND_RATE_HZ)
        dt_log = 1.0 / max(1e-3, ODRIVE_LOG_RATE_HZ)
        last_cmd = time.perf_counter()
        last_log = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            st = self.state.get_state()
            sv = self.state.get_state_version()
            if sv != self._applied_state_version:
                await self._apply_state(st)
                self._applied_state_version = sv
            if self._drivers and st == "enable" and (now - last_cmd) >= dt_cmd:
                positions = self.state.get_pos_cmd()
                for i, (aid, drv) in enumerate(self._drivers):
                    try:
                        drv.set_input_pos(float(positions[i]))
                    except Exception as e:
                        logger.error(f"[ODRV] set_input_pos failed: {e}")
                last_cmd = now
            if (now - last_log) >= dt_log:
                logger.info(f"[ODRV] streaming {len(self._drivers)} axes, state={st}")
                last_log = now
            await asyncio.sleep(0.002)

    async def _run_async(self):
        await self._start_all()
        await self._stream_positions()

    def run(self):
        try:
            asyncio.run(self._run_async())
        except Exception as e:
            logger.error(f"[ODRV] loop error: {e}")


def udp_telemetry_sender(state: RobotState, udp_sock, controller_addr, stop_event):
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
