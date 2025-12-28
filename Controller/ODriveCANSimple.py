import can
import struct
import threading
import enum


class AxisState(enum.IntEnum):
    IDLE = 1
    CLOSED_LOOP_CONTROL = 8
    # ... add others as needed ...


class ODriveAxis:
    def __init__(self, axis_id, manager):
        self.axis_id = axis_id
        self.manager = manager
        self.callbacks = {
            "encoder": None,
            "bus": None,
            "iq": None,
            "heartbeat": None,
            "error": None,
            "temp": None,
        }

    # ---------------- Commands ----------------
    def set_axis_state(self, state: AxisState):
        self.manager._send(self.axis_id, 0x07, int(state).to_bytes(4, "little"))

    def set_input_pos(self, pos_turns, vel_turns=0.0, torque=0.0):
        payload = (
            struct.pack("<f", pos_turns)
            + int(vel_turns * 1000).to_bytes(2, "little", signed=True)
            + int(torque * 1000).to_bytes(2, "little", signed=True)
        )
        self.manager._send(self.axis_id, 0x0C, payload)

    def set_input_vel(self, vel_turns, torque=0.0):
        payload = struct.pack("<ff", vel_turns, torque)
        self.manager._send(self.axis_id, 0x0D, payload)

    def set_input_torque(self, torque):
        payload = struct.pack("<f", torque)
        self.manager._send(self.axis_id, 0x0E, payload)

    # ---------------- Requests ----------------
    def request_encoder_estimates(self):
        self.manager._send(self.axis_id, 0x09, b"", rtr=True)

    def request_bus_measurements(self):
        self.manager._send(self.axis_id, 0x17, b"", rtr=True)

    def request_iq(self):
        self.manager._send(self.axis_id, 0x14, b"", rtr=True)

    def request_temp(self):
        self.manager._send(self.axis_id, 0x15, b"", rtr=True)

    def request_heartbeat(self):
        self.manager._send(self.axis_id, 0x01, b"", rtr=True)

    def request_error(self):
        # errors come inside heartbeat, but allow explicit request
        self.manager._send(self.axis_id, 0x01, b"", rtr=True)

    # ---------------- Callbacks ----------------
    def on_encoder(self, cb):  # cb(pos, vel)
        self.callbacks["encoder"] = cb

    def on_bus(self, cb):  # cb(vbus, ibus)
        self.callbacks["bus"] = cb

    def on_iq(self, cb):  # cb(iq_set, iq_meas)
        self.callbacks["iq"] = cb

    def on_heartbeat(self, cb):  # cb(axis_error, axis_state, ctrl_status)
        self.callbacks["heartbeat"] = cb

    def on_error(self, cb):  # cb(axis_error)
        self.callbacks["error"] = cb

    def on_temp(self, cb):  # cb(fet_temp, motor_temp)
        self.callbacks["temp"] = cb

    # ---------------- Dispatch ----------------
    def _handle_frame(self, cmd_id, data):
        if cmd_id == 0x01:  # Heartbeat
            axis_error = int.from_bytes(data[0:4], "little")
            axis_state = data[4]
            ctrl_status = data[5]
            if self.callbacks["heartbeat"]:
                self.callbacks["heartbeat"](axis_error, axis_state, ctrl_status)
            if axis_error != 0 and self.callbacks["error"]:
                self.callbacks["error"](axis_error)

        elif cmd_id == 0x09 and self.callbacks["encoder"]:
            pos, vel = struct.unpack("<ff", data)
            self.callbacks["encoder"](pos, vel)

        elif cmd_id == 0x14 and self.callbacks["iq"]:
            iq_set, iq_meas = struct.unpack("<ff", data)
            self.callbacks["iq"](iq_set, iq_meas)

        elif cmd_id == 0x17 and self.callbacks["bus"]:
            vbus, ibus = struct.unpack("<ff", data)
            self.callbacks["bus"](vbus, ibus)

        elif cmd_id == 0x15 and self.callbacks["temp"]:
            fet_temp, motor_temp = struct.unpack("<ff", data)
            self.callbacks["temp"](fet_temp, motor_temp)


class ODriveCanManager:
    def __init__(self, canbus="can0"):
        self.bus = can.interface.Bus(channel=canbus, bustype="socketcan")
        self.axes = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listener, daemon=True)
        self._thread.start()

    def add_axis(self, axis_id):
        axis = ODriveAxis(axis_id, self)
        self.axes[axis_id] = axis
        return axis

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.bus.shutdown()

    def _send(self, axis_id, cmd_id, payload, rtr=False):
        arb_id = axis_id * 0x20 + cmd_id
        msg = can.Message(
            arbitration_id=arb_id,
            data=payload,
            is_extended_id=False,
            is_remote_frame=rtr,
        )
        self.bus.send(msg)

    def _listener(self):
        while not self._stop.is_set():
            msg = self.bus.recv(timeout=0.1)
            if not msg:
                continue
            axis_id = msg.arbitration_id // 0x20
            cmd_id = msg.arbitration_id % 0x20
            if axis_id in self.axes:
                self.axes[axis_id]._handle_frame(cmd_id, msg.data)
