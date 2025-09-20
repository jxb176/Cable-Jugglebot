import sys
import threading
import math
import time
import random
from queue import Queue
import socket
import json
import os
import csv

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
+# Telemetry connection timeout (seconds)
+TELEMETRY_TIMEOUT = 2.0

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

    def run(self):
        last_cmd = None
        while not self._stop.is_set():
            try:
                # Connect (block/retry)
                if self.status_cb:
                    self.status_cb("Connecting to robot (TCP)...")
                self._sock = socket.create_connection((self.host, self.port), timeout=5)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self.status_cb:
                    self.status_cb("Connected (TCP)")
                # If we already have a last command, send it after reconnect
                if last_cmd is not None:
                    self._send_cmd(last_cmd)

                # Main send loop
                while not self._stop.is_set():
                    cmd = self.cmd_queue.get()  # blocks until new command
                    last_cmd = cmd
                    self._send_cmd(cmd)
            except Exception as e:
                if self.status_cb:
                    self.status_cb(f"TCP disconnected: {e}. Reconnecting in 1s...")
                self._close()
                time.sleep(1)

        self._close()

    def _send_cmd(self, cmd_value):
        if not self._sock:
            return
        # Only accept dict commands (axes/state). Ignore non-dicts.
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
    """Listen for UDP telemetry and push dict into telem_queue."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Reuse addr for quick restarts
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", udp_port))
    if status_cb:
        status_cb(f"Telemetry: listening UDP :{udp_port}")
    while True:
        try:
            data, _ = sock.recvfrom(4096)
            # New format: {"t": <unix_time>, "pos": [6], "vel": [6]}
            # Legacy: {"t": <unix_time>, "val": <float>}
            telem = json.loads(data.decode("utf-8"))
            telem_queue.put(telem)
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

        # Live feedback (pos/vel) labels
        fb_layout = QVBoxLayout()
        self.fb_labels = []
        for i in range(6):
            row = QHBoxLayout()
            lbl = QLabel(f"A{i+1} pos:")
            val_pos = QLabel("--")
            lbl2 = QLabel("vel:")
            val_vel = QLabel("--")
            row.addWidget(lbl)
            row.addWidget(val_pos)
            row.addWidget(lbl2)
            row.addWidget(val_vel)
            fb_layout.addLayout(row)
            self.fb_labels.append((val_pos, val_vel))
        layout.addLayout(fb_layout)

        # Profile controls: dropdown + send + start + rate
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
        self.profile_refresh_btn.clicked.connect(self.populate_profile_dropdown)
        self.profile_send_btn.clicked.connect(self.on_send_profile)
        self.profile_start_btn.clicked.connect(self.on_start_profile)
        prof_layout.addWidget(QLabel("Profile CSV:"))
        prof_layout.addWidget(self.profile_combo, 1)
        prof_layout.addWidget(self.profile_refresh_btn)
        prof_layout.addWidget(self.profile_send_btn)
        prof_layout.addWidget(QLabel("Rate (Hz):"))
        prof_layout.addWidget(self.profile_rate)
        prof_layout.addWidget(self.profile_start_btn)
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
        self.last_pos = [0.0]*6
        self.last_vel = [0.0]*6
        self.last_telem_time = 0.0

        # Initialize profile dropdown (Profiles subfolder)
        self.populate_profile_dropdown()

        # Timer to refresh GUI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)  # update every 100 ms

        # Timer to check telemetry connection status
        self.conn_timer = QTimer()
        self.conn_timer.timeout.connect(self.check_connection_status)
        self.conn_timer.start(500)  # check every 0.5s

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

    def update_gui(self):
        """Check telemetry and update GUI."""
        while not self.telem_queue.empty():
            telem = self.telem_queue.get()
            self.last_telem_time = time.time()

            # --- Dict telemetry (new format) ---
            if isinstance(telem, dict):
                t = float(telem.get("t", time.time()))

                # Array format: {"pos":[...], "vel":[...]}
                if "pos" in telem or "vel" in telem:
                    pos = telem.get("pos", self.last_pos)
                    vel = telem.get("vel", self.last_vel)

                    # Update feedback labels (None -> '---')
                    for i in range(6):
                        p = pos[i] if (isinstance(pos, list) and i < len(pos)) else None
                        v = vel[i] if (isinstance(vel, list) and i < len(vel)) else None
                        p_text = "---" if p is None else f"{float(p):.4f}"
                        v_text = "---" if v is None else f"{float(v):.4f}"
                        self.fb_labels[i][0].setText(p_text)
                        self.fb_labels[i][1].setText(v_text)

                    # Cache last arrays (None -> NaN)
                    if isinstance(pos, list) and len(pos) >= 6:
                        self.last_pos = [(float(x) if x is not None else float("nan")) for x in pos[:6]]
                    if isinstance(vel, list) and len(vel) >= 6:
                        self.last_vel = [(float(x) if x is not None else float("nan")) for x in vel[:6]]

                    # Plot axis 1 pos
                    self.xdata.append(t - self.start_time)
                    a1 = self.last_pos[0] if self.last_pos and self.last_pos[0] == self.last_pos[0] else 0.0
                    self.ydata.append(a1)
                    if len(self.xdata) > 200:
                        self.xdata = self.xdata[-200:]
                        self.ydata = self.ydata[-200:]

                # Legacy single value dict: {"val": <float>}
                elif "val" in telem:
                    val = float(telem["val"])
                    self.xdata.append(t - self.start_time)
                    self.ydata.append(val)
                    if len(self.xdata) > 200:
                        self.xdata = self.xdata[-200:]
                        self.ydata = self.ydata[-200:]

            # --- Legacy tuple format: (t, val) ---
            elif isinstance(telem, (tuple, list)) and len(telem) >= 2:
                try:
                    t = float(telem[0])
                    val = float(telem[1])
                except Exception:
                    continue
                self.xdata.append(t - self.start_time)
                self.ydata.append(val)
                if len(self.xdata) > 200:
                    self.xdata = self.xdata[-200:]
                    self.ydata = self.ydata[-200:]

        # --- Update plot ---
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