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
        self.callbacks = {"encoder": None, "bus": None, "iq": None}

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

    def request_encoder_estimates(self):
        self.manager._send(self.axis_id, 0x09, b"", rtr=True)

    def request_bus_measurements(self):
        self.manager._send(self.axis_id, 0x17, b"", rtr=True)

    # ---------------- Callbacks ----------------
    def on_encoder(self, cb):  # cb(pos, vel)
        self.callbacks["encoder"] = cb

    def on_bus(self, cb):  # cb(vbus, ibus)
        self.callbacks["bus"] = cb

    def on_iq(self, cb):  # cb(iq_set, iq_meas)
        self.callbacks["iq"] = cb

    # ---------------- Dispatch ----------------
    def _handle_frame(self, cmd_id, data):
        if cmd_id == 0x09 and self.callbacks["encoder"]:
            pos, vel = struct.unpack("<ff", data)
            self.callbacks["encoder"](pos, vel)
        elif cmd_id == 0x14 and self.callbacks["iq"]:
            iq_set, iq_meas = struct.unpack("<ff", data)
            self.callbacks["iq"](iq_set, iq_meas)
        elif cmd_id == 0x17 and self.callbacks["bus"]:
            vbus, ibus = struct.unpack("<ff", data)
            self.callbacks["bus"](vbus, ibus)


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
