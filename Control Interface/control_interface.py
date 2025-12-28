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
    QLabel, QSlider, QHBoxLayout, QDoubleSpinBox, QComboBox,
    QTableWidget, QTableWidgetItem, QSizePolicy
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

        # ---- Consistent per-axis color coding (A1..A6) ----
        # Use a fixed palette so colors stay consistent across plots
        self.axis_colors = [
            (255, 80, 80),  # A1 - red
            (80, 200, 120),  # A2 - green
            (80, 160, 255),  # A3 - blue
            (255, 200, 80),  # A4 - amber
            (190, 120, 255),  # A5 - purple
            (80, 220, 220),  # A6 - cyan
        ]

        def axis_pen(i: int, width: int = 2, style=Qt.PenStyle.SolidLine):
            return pg.mkPen(color=self.axis_colors[i], width=width, style=style)

        # History length (number of samples kept in plots)
        self.history_seconds = 20.0

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
            spin.valueChanged.connect(self.send_axes)  # Send axes command on change
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

        # Profile controls
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

        # ---- Axis numeric matrix (6 rows x columns) ----
        self.axis_table_cols = [
            "State",
            "Error",
            "Pos (turns)",
            "Vel (turns/s)",
            "Motor I (A)",
            "Bus V (V)",
            "Bus I (A)",
            "Temp Motor (°C)",
            "Temp FET (°C)",
        ]

        self.axis_table = QTableWidget(6, len(self.axis_table_cols))
        self.axis_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Make the table tall enough to show all 6 rows + header
        row_h = self.axis_table.verticalHeader().defaultSectionSize()
        hdr_h = self.axis_table.horizontalHeader().height()
        frame = 2 * self.axis_table.frameWidth()
        self.axis_table.setMinimumHeight(hdr_h + 6 * row_h + frame + 8)

        self.axis_table.setHorizontalHeaderLabels(self.axis_table_cols)
        self.axis_table.setVerticalHeaderLabels([f"A{i+1}" for i in range(6)])
        self.axis_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.axis_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        # optional: nicer sizing
        #self.axis_table.horizontalHeader().setStretchLastSection(True)
        self.axis_table.resizeColumnsToContents()

        layout.addWidget(QLabel("Axis Data"))
        layout.addWidget(self.axis_table)


        # --- Telemetry plots ---
        # Position plot (6 traces)
        self.plot_pos = pg.PlotWidget(title="Position (turns) — A1..A6")
        self.plot_pos.setLabel("bottom", "Time", "s")
        self.plot_pos.setLabel("left", "Position", "turns")
        self.plot_pos.showGrid(x=True, y=True)
        self.plot_pos.addLegend()
        self.curves_pos = []
        for i in range(6):
            c = self.plot_pos.plot(name=f"A{i + 1}", pen=axis_pen(i, width=2))
            self.curves_pos.append(c)
        layout.addWidget(self.plot_pos)

        # Velocity plot (6 traces)
        self.plot_vel = pg.PlotWidget(title="Velocity (turns/s) — A1..A6")
        self.plot_vel.setLabel("bottom", "Time", "s")
        self.plot_vel.setLabel("left", "Velocity", "turns/s")
        self.plot_vel.showGrid(x=True, y=True)
        self.plot_vel.addLegend()
        self.curves_vel = []
        for i in range(6):
            c = self.plot_vel.plot(name=f"A{i + 1}", pen=axis_pen(i, width=2))
            self.curves_vel.append(c)
        layout.addWidget(self.plot_vel)

        # Temperature plot (12 traces: motor+fet for each axis)
        self.plot_temp = pg.PlotWidget(title="Temperatures (°C) — Motor + FET (A1..A6)")
        self.plot_temp.setLabel("bottom", "Time", "s")
        self.plot_temp.setLabel("left", "Temp", "°C")
        self.plot_temp.showGrid(x=True, y=True)
        self.plot_temp.addLegend()
        self.curves_temp_motor = []
        self.curves_temp_fet = []
        for i in range(6):
            self.curves_temp_motor.append(
                self.plot_temp.plot(name=f"A{i + 1} Motor", pen=axis_pen(i, width=2, style=Qt.PenStyle.SolidLine))
            )
            self.curves_temp_fet.append(
                self.plot_temp.plot(name=f"A{i + 1} FET", pen=axis_pen(i, width=2, style=Qt.PenStyle.DashLine))
            )
        layout.addWidget(self.plot_temp)

        # Bus Voltage plot
        self.busv_plot = pg.PlotWidget(title="Bus Voltage (V)")
        self.busv_plot.setLabel('bottom', 'Time', 's')
        self.busv_plot.setLabel('left', 'Voltage', 'V')
        self.busv_plot.showGrid(x=True, y=True)
        self.busv_plot.addLegend()
        self.curves_busv = []
        for i in range(6):
            self.curves_busv.append(self.busv_plot.plot(name=f"A{i + 1}", pen=axis_pen(i, width=2)))
        layout.addWidget(self.busv_plot)

        # Current plot (12 traces: motor current + bus current for each axis)
        self.plot_cur = pg.PlotWidget(title="Currents (A) — Motor + Bus (A1..A6)")
        self.plot_cur.setLabel("bottom", "Time", "s")
        self.plot_cur.setLabel("left", "Current", "A")
        self.plot_cur.showGrid(x=True, y=True)
        self.plot_cur.addLegend()
        self.curves_cur_motor = []
        self.curves_cur_bus = []
        for i in range(6):
            self.curves_cur_motor.append(
                self.plot_cur.plot(name=f"A{i + 1} Motor I", pen=axis_pen(i, width=2, style=Qt.PenStyle.SolidLine))
            )
            self.curves_cur_bus.append(
                self.plot_cur.plot(name=f"A{i + 1} Bus I", pen=axis_pen(i, width=2, style=Qt.PenStyle.DashLine))
            )
        layout.addWidget(self.plot_cur)


        self.setLayout(layout)

        # Data buffers
        self.tbuf = []

        self.pos_buf = [[] for _ in range(6)]
        self.vel_buf = [[] for _ in range(6)]

        self.bus_v_buf = [[] for _ in range(6)]

        self.cur_motor_buf = [[] for _ in range(6)]
        self.cur_bus_buf = [[] for _ in range(6)]

        self.temp_motor_buf = [[] for _ in range(6)]
        self.temp_fet_buf = [[] for _ in range(6)]

        # --- Heartbeat / state / error buffers (ints) ---
        self.axis_state_buf = [[] for _ in range(6)]
        self.axis_error_buf = [[] for _ in range(6)]

        self.start_time = time.time()
        self.last_pos = [0.0]*6
        self.last_vel = [0.0]*6
        self.last_telem_time = 0
        self.telem_timeout = 2.0  # seconds

        # Initialize profile dropdown
        self.populate_profile_dropdown()

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)  # update every 100 ms

        self.conn_timer = QTimer()
        self.conn_timer.timeout.connect(self.check_connection_status)
        self.conn_timer.start(500)  # check every 0.5 s

    def check_connection_status(self):
        now = time.time()
        if self.last_telem_time == 0 or (now - self.last_telem_time) > self.telem_timeout:
            self.status_label.setText("⚠️ Waiting for telemetry…")
        else:
            self.status_label.setText("✅ Connected (telemetry streaming)")

    def _profiles_dir(self) -> str:
        base = os.getcwd()
        pdir = os.path.join(base, "Profiles")
        os.makedirs(pdir, exist_ok=True)
        return pdir

    def send_axes(self, *_):
        positions = [float(sp.value()) for sp in self.axis_spins]
        cmd = {"type": "axes", "positions": positions, "units": "turns"}
        _queue_put_latest(self.cmd_queue, cmd)

    def send_state(self, state_value: str):
        cmd = {"type": "state", "value": state_value}
        _queue_put_latest(self.cmd_queue, cmd)

    def _load_csv_as_profile(self, path: str):
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(cell.strip() for cell in r)]
        if not rows:
            raise ValueError("empty CSV")
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
        times = [row[0] for row in profile_rows]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("time column must be non-decreasing")
        return profile_rows

    def populate_profile_dropdown(self):
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
        rate = float(self.profile_rate.value())
        cmd = {"type": "profile_start", "rate_hz": rate}
        _queue_put_latest(self.cmd_queue, cmd)
        self.status_label.setText(f"Profile start requested at {rate:.1f} Hz")

    def _append_vec6(self, buf_list, vec):
        """Append a length-6 vector to per-axis buffers. Missing/None -> NaN."""
        if not isinstance(vec, list):
            vec = []
        for i in range(6):
            v = vec[i] if i < len(vec) else None
            if v is None:
                buf_list[i].append(float("nan"))
            else:
                try:
                    buf_list[i].append(float(v))
                except Exception:
                    buf_list[i].append(float("nan"))

    def _append_vec6_int(self, buf_list, vec):
        """Append a length-6 vector to per-axis buffers. Missing/None -> None."""
        if not isinstance(vec, list):
            vec = []
        for i in range(6):
            v = vec[i] if i < len(vec) else None
            if v is None:
                buf_list[i].append(None)
            else:
                try:
                    buf_list[i].append(int(v))
                except Exception:
                    buf_list[i].append(None)

    def _trim_history(self):
        if not self.tbuf:
            return
        t_latest = self.tbuf[-1]
        t_min = t_latest - self.history_seconds

        # find first index to keep
        k0 = 0
        for k in range(len(self.tbuf)):
            if self.tbuf[k] >= t_min:
                k0 = k
                break

        if k0 <= 0:
            return

        self.tbuf = self.tbuf[k0:]
        for banks in (
                self.pos_buf, self.vel_buf,
                self.temp_motor_buf, self.temp_fet_buf,
                self.cur_motor_buf, self.cur_bus_buf,
                self.bus_v_buf,
                self.axis_state_buf, self.axis_error_buf,
        ):
            for i in range(6):
                banks[i] = banks[i][k0:]

    def _set_table_cell(self, row: int, col: int, value, fmt=None):

        """
        value can be float/int/None/NaN/str. If fmt is provided, it's used for numeric formatting.
        """
        # --- format text ---
        if value is None:
            text = "---"
        elif isinstance(value, str):
            text = value
        else:
            try:
                v = float(value)
                if v != v:  # NaN
                    text = "---"
                else:
                    text = (fmt.format(v) if fmt else str(v))
            except Exception:
                text = "---"

        item = self.axis_table.item(row, col)
        if item is None:
            item = QTableWidgetItem(text)
            # Right align data cells
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.axis_table.setItem(row, col, item)
        else:
            item.setText(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def update_gui(self):
        updated = False

        while not self.telem_queue.empty():
            telem = self.telem_queue.get()
            if not isinstance(telem, dict):
                continue

            self.last_telem_time = time.time()
            t = float(telem.get("t", time.time()))
            trel = t - self.start_time

            # timebase
            self.tbuf.append(trel)

            # required arrays
            self._append_vec6(self.pos_buf, telem.get("pos", []))
            self._append_vec6(self.vel_buf, telem.get("vel", []))

            # temps (match your keys)
            self._append_vec6(self.temp_fet_buf, telem.get("temp_fet", []))
            self._append_vec6(self.temp_motor_buf, telem.get("temp_motor", []))

            # currents
            self._append_vec6(self.cur_bus_buf, telem.get("bus_i", []))
            # motor current is optional; will plot NaNs until you send it
            self._append_vec6(self.cur_motor_buf, telem.get("motor_i", []))

            #bus voltage
            self._append_vec6(self.bus_v_buf, telem.get("bus_v", []))

            # heartbeat/state/error (match your telemetry keys)
            # Expecting vectors like: "axis_state": [6], "axis_error": [6]
            self._append_vec6_int(self.axis_state_buf, telem.get("axis_state", []))
            self._append_vec6_int(self.axis_error_buf, telem.get("axis_error", []))

            self._trim_history()
            updated = True

        if not updated:
            return

        # ---- Update numeric matrix using newest samples ----
        for i in range(6):
            state_i = self.axis_state_buf[i][-1] if self.axis_state_buf[i] else None
            error_i = self.axis_error_buf[i][-1] if self.axis_error_buf[i] else None

            pos_i = self.pos_buf[i][-1] if self.pos_buf[i] else float("nan")
            vel_i = self.vel_buf[i][-1] if self.vel_buf[i] else float("nan")

            motor_i = self.cur_motor_buf[i][-1] if self.cur_motor_buf[i] else float("nan")
            bus_i   = self.cur_bus_buf[i][-1] if self.cur_bus_buf[i] else float("nan")
            busv_i = self.bus_v_buf[i][-1] if self.bus_v_buf[i] else float("nan")

            tm_i = self.temp_motor_buf[i][-1] if self.temp_motor_buf[i] else float("nan")
            tf_i = self.temp_fet_buf[i][-1] if self.temp_fet_buf[i] else float("nan")

            # Cols: State, Error, Pos, Vel, MotorI, BusV, BusI, TempMotor, TempFET
            self._set_table_cell(i, 0, state_i, None)
            self._set_table_cell(i, 1, error_i, None)
            self._set_table_cell(i, 2, pos_i, "{:.4f}")
            self._set_table_cell(i, 3, vel_i, "{:.4f}")
            self._set_table_cell(i, 4, motor_i, "{:.2f}")
            self._set_table_cell(i, 5, busv_i, "{:.2f}")
            self._set_table_cell(i, 6, bus_i, "{:.2f}")
            self._set_table_cell(i, 7, tm_i, "{:.1f}")
            self._set_table_cell(i, 8, tf_i, "{:.1f}")

        # update plots
        x = self.tbuf
        for i in range(6):
            self.curves_pos[i].setData(x, self.pos_buf[i])
            self.curves_vel[i].setData(x, self.vel_buf[i])

            self.curves_temp_motor[i].setData(x, self.temp_motor_buf[i])
            self.curves_temp_fet[i].setData(x, self.temp_fet_buf[i])

            self.curves_cur_bus[i].setData(x, self.cur_bus_buf[i])
            self.curves_cur_motor[i].setData(x, self.cur_motor_buf[i])

            self.curves_busv[i].setData(x, self.bus_v_buf[i])

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