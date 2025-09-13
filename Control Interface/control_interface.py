import sys
import threading
import time
import random
from queue import Queue
import socket
import json

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QSlider, QHBoxLayout, QDoubleSpinBox
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
        # Support dict commands for axes/state; fallback to legacy numeric speed
        if isinstance(cmd_value, dict):
            payload = cmd_value
        else:
            try:
                payload = {"type": "speed", "value": float(cmd_value)}
            except Exception:
                return
        msg = json.dumps(payload) + "\n"
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

        # Stop button
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.clicked.connect(lambda: self.send_command(0))
        layout.addWidget(self.stop_btn)

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

        # Timer to refresh GUI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)  # update every 100 ms

    def send_command(self, value):
        """Send a command to the robot (via TCP command thread)."""
        _queue_put_latest(self.cmd_queue, value)

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

        # Update plot
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
