# robot_server.py
import socket
import json
import time
import threading
import random
import os
import logging
import base64
from datetime import datetime

TCP_CMD_PORT = 5555
UDP_TELEM_PORT = 5556

# Initialize logging to Logs folder
def _init_logging():
    logs_dir = os.path.join(os.getcwd(), "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"robot_{ts}.log")
    logger = logging.getLogger("robot")
    logger.setLevel(logging.INFO)
    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path

logger, LOG_FILE_PATH = _init_logging()

class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.controller_ip = None  # set when TCP client connects
        # 6-axis targets (turns) and robot state
        self.axes = [0.0] * 6         # positions in turns
        self.state = "disable"        # "enable" | "disable" | "estop"
        # Profile storage and player
        self.profile = []             # list of tuples: (t, [6 positions])
        self.player_thread = None     # type: ProfilePlayer | None

    def set_controller_ip(self, ip):
        with self.lock:
            self.controller_ip = ip
        logger.info(f"Controller IP set to {ip}")

    def get_controller_ip(self):
        with self.lock:
            return self.controller_ip

    # Axes and state
    def set_axes(self, positions):
        if not isinstance(positions, (list, tuple)) or len(positions) != 6:
            raise ValueError("positions must be length-6 list/tuple")
        with self.lock:
            self.axes = [float(x) for x in positions]
        logger.info("Axes set to: " + ", ".join(f"{x:.4f}" for x in self.axes))

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

    # Profile management
    def set_profile(self, profile_points):
        """profile_points: list of [t, a1..a6]"""
        if not isinstance(profile_points, (list, tuple)) or len(profile_points) == 0:
            raise ValueError("profile must be a non-empty list")
        prof = []
        for row in profile_points:
            if not isinstance(row, (list, tuple)) or len(row) < 7:
                raise ValueError("each profile row must be [t, a1..a6]")
            t = float(row[0])
            axes = [float(x) for x in row[1:7]]
            prof.append((t, axes))
        # Ensure non-decreasing time
        times = [p[0] for p in prof]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("profile time column must be non-decreasing")
        with self.lock:
            self.profile = prof
        logger.info(f"Profile uploaded: {len(prof)} points, duration {prof[-1][0] - prof[0][0]:.3f}s")

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
    def __init__(self, state: RobotState, profile, rate_hz: float):
        super().__init__(daemon=True)
        self.state = state
        self.profile = profile  # list[(t, [6])]
        self.dt = 1.0 / max(1e-3, float(rate_hz))
        self._stop = threading.Event()

        # Pre-normalize time to start at zero
        t0 = self.profile[0][0]
        self.norm_profile = [(t - t0, axes) for (t, axes) in self.profile]
        self.duration = self.norm_profile[-1][0]

    def stop(self):
        self._stop.set()

    def run(self):
        logger.info(f"[PROFILE] Starting playback at {1.0/self.dt:.1f} Hz, duration {self.duration:.3f}s")
        start = time.perf_counter()
        k = 0  # segment index: between norm_profile[k] and [k+1]
        while not self._stop.is_set():
            t = time.perf_counter() - start
            if t >= self.duration:
                # Clamp to final point and finish
                self.state.set_axes(self.norm_profile[-1][1])
                logger.info("[PROFILE] Completed")
                break

            # Advance segment to bracket current time
            while k + 1 < len(self.norm_profile) and self.norm_profile[k + 1][0] < t:
                k += 1
            # Find the two points for interpolation
            t0, p0 = self.norm_profile[k]
            t1, p1 = self.norm_profile[min(k + 1, len(self.norm_profile) - 1)]
            if t1 <= t0:
                alpha = 0.0
            else:
                alpha = (t - t0) / (t1 - t0)
                if alpha < 0.0:
                    alpha = 0.0
                elif alpha > 1.0:
                    alpha = 1.0
            # Linear interpolate each axis
            axes = [p0[i] + alpha * (p1[i] - p0[i]) for i in range(6)]
            self.state.set_axes(axes)

            # Sleep to next tick
            time.sleep(self.dt)
        # Clear active player reference if we are the current one
        with self.state.lock:
            if self.state.player_thread is self:
                self.state.player_thread = None


def _send_log_over_tcp(conn):
    """Read current log file and send as one base64 JSON line."""
    try:
        with open(LOG_FILE_PATH, "rb") as f:
            data = f.read()
        payload = {
            "type": "log_file",
            "filename": os.path.basename(LOG_FILE_PATH),
            "data_b64": base64.b64encode(data).decode("ascii"),
        }
        conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        logger.info(f"Sent log file to controller: {payload['filename']} ({len(data)} bytes)")
    except Exception as e:
        logger.error(f"Failed to send log: {e}")

def tcp_command_server(state: RobotState):
    """Accepts a single TCP client, receives newline-delimited JSON commands."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", TCP_CMD_PORT))
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
                            positions = msg.get("positions", [])
                            state.set_axes(positions)
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
                        elif mtype == "log_request":
                            _send_log_over_tcp(conn)
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
    """Sends telemetry to the controller IP over UDP at 10 Hz."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        time.sleep(0.1)
        ctrl_ip = state.get_controller_ip()
        if not ctrl_ip:
            continue
        # Example telemetry: derive a single value from axis 1 plus small noise
        axes = state.get_axes()
        a1 = axes[0] if axes else 0.0
        val = a1 + random.uniform(-0.05, 0.05)
        msg = {"t": time.time(), "val": float(val)}
        try:
            sock.sendto(json.dumps(msg).encode("utf-8"), (ctrl_ip, UDP_TELEM_PORT))
        except Exception as e:
            logger.error(f"[UDP] Telemetry send error: {e}")

def axes_state_logger(state: RobotState):
    """Prints current robot state and 6-axis targets at 1 Hz."""
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
    threading.Thread(target=tcp_command_server, args=(state,), daemon=True).start()
    threading.Thread(target=udp_telemetry_sender, args=(state,), daemon=True).start()
    threading.Thread(target=axes_state_logger, args=(state,), daemon=True).start()
    logger.info("Robot server running. Press Ctrl+C to exit.")
    while True:
        time.sleep(1)

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QSlider, QHBoxLayout, QDoubleSpinBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer

import pyqtgraph as pg


# Networking configuration
ROBOT_HOST = "jugglepi.local"  # <-- set to your Raspberry Pi IP or hostname
TCP_CMD_PORT = 5555
UDP_TELEM_PORT = 5556


def _queue_put_latest(q: Queue, item):
    """Keep only the newest item in the queue."""
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass
    q.put(item)


class CommandClient(threading.Thread):
    """TCP client that reliably sends commands to the robot with auto-reconnect."""
    def __init__(self, host, port, cmd_queue: Queue, status_cb=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.cmd_queue = cmd_queue
        self.status_cb = status_cb
        self._stop = threading.Event()
        self._sock = None
        self._rx_thread = None

    def run(self):
        last_cmd = None
        while not self._stop.is_set():
            try:
                if self.status_cb:
                    self.status_cb("Connecting to robot (TCP)...")
                self._sock = socket.create_connection((self.host, self.port), timeout=5)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self.status_cb:
                    self.status_cb("Connected (TCP)")

                # Start receiver thread
                self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True)
                self._rx_thread.start()

                if last_cmd is not None:
                    self._send_cmd(last_cmd)

                while not self._stop.is_set():
                    cmd = self.cmd_queue.get()
                    last_cmd = cmd
                    self._send_cmd(cmd)
            except Exception as e:
                if self.status_cb:
                    self.status_cb(f"TCP disconnected: {e}. Reconnecting in 1s...")
                self._close()
                time.sleep(1)
        self._close()

    def _recv_loop(self):
        try:
            f = self._sock.makefile("r")
            for line in f:
                try:
                    msg = json.loads(line.strip())
                    mtype = msg.get("type")
                    if mtype == "log_file":
                        self._handle_log_file(msg)
                    else:
                        # Other server-originated messages could be handled here
                        pass
                except Exception as e:
                    if self.status_cb:
                        self.status_cb(f"RX parse error: {e}")
        except Exception as e:
            if self.status_cb:
                self.status_cb(f"RX error: {e}")

    def _handle_log_file(self, msg: dict):
        try:
            fname = msg.get("filename", "robot.log")
            data_b64 = msg.get("data_b64", "")
            data = base64.b64decode(data_b64.encode("ascii"))
            logs_dir = os.path.join(os.getcwd(), "Logs")
            os.makedirs(logs_dir, exist_ok=True)
            out_path = os.path.join(logs_dir, fname)
            with open(out_path, "wb") as f:
                f.write(data)
            if self.status_cb:
                self.status_cb(f"Saved log to {out_path}")
        except Exception as e:
            if self.status_cb:
                self.status_cb(f"Failed to save log: {e}")

    def _send_cmd(self, cmd_value):
        if not self._sock:
            return
        if not isinstance(cmd_value, dict):
            return
        msg = json.dumps(cmd_value) + "\n"
        self._sock.sendall(msg.encode("utf-8"))

    def stop(self):
        self._stop.set()
        self._close()

    def _close(self):
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None


def telemetry_listener(udp_port: int, telem_queue: Queue, status_cb=None):
    """Listen for UDP telemetry and push (t, val) into telem_queue."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Reuse addr for quick restarts
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", udp_port))
    if status_cb:
        status_cb(f"Telemetry: listening UDP :{udp_port}")
    while True:
        try:
            data, _ = sock.recvfrom(4096)
            # Expect JSON: {"t": <unix_time>, "val": <float>}
            telem = json.loads(data.decode("utf-8"))
            t = float(telem.get("t", time.time()))
            val = float(telem["val"])
            telem_queue.put((t, val))
        except Exception as e:
            if status_cb:
                status_cb(f"Telemetry error: {e}")
            # brief pause to avoid tight loop on persistent error
            time.sleep(0.05)


class RobotGUI(QWidget):
    def __init__(self, cmd_queue, telem_queue):
        super().__init__()
        self.setWindowTitle("Robot Controller + Telemetry")
        self.resize(800, 600)

        self.cmd_queue = cmd_queue
        self.telem_queue = telem_queue

        # --- Layout ---
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Telemetry: waiting...")
        layout.addWidget(self.status_label)

        # Axis controls (1-6) in turns
        axes_layout = QVBoxLayout()
        self.axis_spins = []
        for i in range(6):
            row = QHBoxLayout()
            lbl = QLabel(f"Axis {i+1} (turns)")
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(-10.0, 10.0)
            spin.setSingleStep(0.01)
            spin.setValue(0.0)
            # Send an axes command on any change
            spin.valueChanged.connect(self.send_axes)
            row.addWidget(lbl)
            row.addWidget(spin)
            axes_layout.addLayout(row)
            self.axis_spins.append(spin)
        layout.addLayout(axes_layout)

        # State controls
        state_layout = QHBoxLayout()
        self.btn_enable = QPushButton("Enable")
        self.btn_disable = QPushButton("Disable")
        self.btn_estop = QPushButton("ESTOP")
        self.btn_enable.clicked.connect(lambda: self.send_state("enable"))
        self.btn_disable.clicked.connect(lambda: self.send_state("disable"))
        self.btn_estop.clicked.connect(lambda: self.send_state("estop"))
        state_layout.addWidget(self.btn_enable)
        state_layout.addWidget(self.btn_disable)
        state_layout.addWidget(self.btn_estop)
        layout.addLayout(state_layout)

        # Profile controls: dropdown + send + start + rate + fetch log
        prof_layout = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_refresh_btn = QPushButton("Refresh")
        self.profile_send_btn = QPushButton("Send Profile")
        self.profile_rate = QDoubleSpinBox()
        self.profile_rate.setDecimals(1)
        self.profile_rate.setRange(1.0, 1000.0)
        self.profile_rate.setSingleStep(10.0)
        self.profile_rate.setValue(100.0)
        self.profile_start_btn = QPushButton("Start Profile")
        self.profile_fetch_log_btn = QPushButton("Fetch Log")
        self.profile_refresh_btn.clicked.connect(self.populate_profile_dropdown)
        self.profile_send_btn.clicked.connect(self.on_send_profile)
        self.profile_start_btn.clicked.connect(self.on_start_profile)
        self.profile_fetch_log_btn.clicked.connect(self.on_fetch_log)
        prof_layout.addWidget(QLabel("Profile CSV:"))
        prof_layout.addWidget(self.profile_combo, 1)
        prof_layout.addWidget(self.profile_refresh_btn)
        prof_layout.addWidget(self.profile_send_btn)
        prof_layout.addWidget(QLabel("Rate (Hz):"))
        prof_layout.addWidget(self.profile_rate)
        prof_layout.addWidget(self.profile_start_btn)
        prof_layout.addWidget(self.profile_fetch_log_btn)
        layout.addLayout(prof_layout)

        # --- Telemetry plot ---
        self.plot = pg.PlotWidget(title="Telemetry Data (Sensor Value)")
        self.plot.setLabel('bottom', 'Time', 's')
        self.plot.setLabel('left', 'Value')
        self.plot.showGrid(x=True, y=True)

        self.curve = self.plot.plot(pen='y')
        layout.addWidget(self.plot)

        self.setLayout(layout)

        # Data buffer
        self.xdata = []
        self.ydata = []
        self.start_time = time.time()

        # Initialize profile dropdown
        self.populate_profile_dropdown()

        # Timer to refresh GUI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)  # update every 100 ms

    def _profiles_dir(self) -> str:
        """Return absolute path to the Profiles subfolder, creating it if missing."""
        base = os.getcwd()
        pdir = os.path.join(base, "Profiles")
        os.makedirs(pdir, exist_ok=True)
        return pdir

    def send_axes(self, *_):
        """Send 6-axis position command in turns."""
        positions = [float(sp.value()) for sp in self.axis_spins]
        cmd = {"type": "axes", "positions": positions, "units": "turns"}
        _queue_put_latest(self.cmd_queue, cmd)

    def send_state(self, state_value: str):
        """Send robot state command."""
        cmd = {"type": "state", "value": state_value}
        _queue_put_latest(self.cmd_queue, cmd)

    def _load_csv_as_profile(self, path: str):
        """Load CSV profile with rows: time, axis1..axis6."""
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(cell.strip() for cell in r)]
        if not rows:
            raise ValueError("empty CSV")
        # Skip header if first cell not numeric
        start_idx = 0
        try:
            float(rows[0][0])
        except Exception:
            start_idx = 1
        profile_rows = []
        for r in rows[start_idx:]:
            if len(r) < 7:
                raise ValueError("each row must have at least 7 columns: time + 6 axes")
            t = float(r[0])
            axes = [float(x) for x in r[1:7]]
            profile_rows.append([t] + axes)
        # Ensure monotonic non-decreasing time
        times = [row[0] for row in profile_rows]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("time column must be non-decreasing")
        return profile_rows

    def populate_profile_dropdown(self):
        """Scan Profiles subdirectory for CSV files and populate the dropdown."""
        pdir = self._profiles_dir()
        csvs = sorted([f for f in os.listdir(pdir) if f.lower().endswith(".csv")])
        self.profile_combo.clear()
        if not csvs:
            self.profile_combo.addItem("(no .csv files in Profiles/)")
            self.profile_combo.setEnabled(False)
        else:
            self.profile_combo.setEnabled(True)
            for f in csvs:
                self.profile_combo.addItem(f)

    def on_send_profile(self):
        """Parse selected CSV from Profiles and send it to the robot server."""
        if not self.profile_combo.isEnabled():
            self.status_label.setText("No CSV profile to send (Profiles/ empty)")
            return
        fname = self.profile_combo.currentText()
        if not fname or fname.startswith("("):
            self.status_label.setText("Select a valid CSV profile")
            return
        path = os.path.join(self._profiles_dir(), fname)
        try:
            profile_rows = self._load_csv_as_profile(path)
            cmd = {"type": "profile_upload", "profile": profile_rows}
            _queue_put_latest(self.cmd_queue, cmd)
            self.status_label.setText(f"Sent profile: {fname} ({len(profile_rows)} pts)")
        except Exception as e:
            self.status_label.setText(f"Profile send failed: {e}")

    def on_start_profile(self):
        """Send a profile_start with selected rate."""
        rate = float(self.profile_rate.value())
        cmd = {"type": "profile_start", "rate_hz": rate}
        _queue_put_latest(self.cmd_queue, cmd)
        self.status_label.setText(f"Profile start requested at {rate:.1f} Hz")

    def on_fetch_log(self):
        """Request the current log file from the robot server."""
        cmd = {"type": "log_request"}
        _queue_put_latest(self.cmd_queue, cmd)
        self.status_label.setText("Log requested from robot")

    def update_gui(self):
        """Check telemetry and update GUI."""
        while not self.telem_queue.empty():
            t, val = self.telem_queue.get()
            self.xdata.append(t - self.start_time)
            self.ydata.append(val)
            # Limit buffer size
            if len(self.xdata) > 200:
                self.xdata = self.xdata[-200:]
                self.ydata = self.ydata[-200:]
            self.status_label.setText(f"Telemetry: value={val:.2f}")
        self.curve.setData(self.xdata, self.ydata)


# --- Simulated Robot ---  # (not used when connected to real robot)
# def robot_sim(cmd_queue, telem_queue):
#     ...

if __name__ == "__main__":
    cmd_queue = Queue(maxsize=1)   # keep only the latest command
    telem_queue = Queue()

    # Start telemetry listener thread (UDP)
    telem_thread = threading.Thread(
        target=telemetry_listener,
        args=(UDP_TELEM_PORT, telem_queue, lambda s: print(s)),
        daemon=True,
    )
    telem_thread.start()

    # Start TCP command client
    cmd_client = CommandClient(
        ROBOT_HOST, TCP_CMD_PORT, cmd_queue, status_cb=lambda s: print(s)
    )
    cmd_client.start()

    # Start GUI
    app = QApplication(sys.argv)
    gui = RobotGUI(cmd_queue, telem_queue)
    gui.show()
    sys.exit(app.exec())
