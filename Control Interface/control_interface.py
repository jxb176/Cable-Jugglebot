import sys
import threading
import time
import random
from queue import Queue

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QSlider, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer

import pyqtgraph as pg


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

        # Control slider
        ctrl_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(-100, 100)
        self.speed_slider.setValue(0)
        self.speed_slider.valueChanged.connect(self.send_command)
        ctrl_layout.addWidget(QLabel("Speed"))
        ctrl_layout.addWidget(self.speed_slider)
        layout.addLayout(ctrl_layout)

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
        """Send a command to the robot (via queue)."""
        self.cmd_queue.put(value)

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


# --- Simulated Robot ---
def robot_sim(cmd_queue, telem_queue):
    """Fake robot that responds to commands and generates telemetry."""
    speed = 0
    t0 = time.time()

    while True:
        # Check if command arrived
        try:
            speed = cmd_queue.get_nowait()
            print(f"[Robot] New speed command: {speed}")
        except:
            pass

        # Generate fake telemetry (value drifts with speed)
        val = random.random() * 2 + 0.1 * speed
        t = time.time() - t0
        telem_queue.put((t, val))

        time.sleep(0.1)  # simulate robot update rate (10 Hz)


if __name__ == "__main__":
    cmd_queue = Queue()
    telem_queue = Queue()

    # Start robot thread
    t = threading.Thread(target=robot_sim, args=(cmd_queue, telem_queue), daemon=True)
    t.start()

    # Start GUI
    app = QApplication(sys.argv)
    gui = RobotGUI(cmd_queue, telem_queue)
    gui.show()
    sys.exit(app.exec())
