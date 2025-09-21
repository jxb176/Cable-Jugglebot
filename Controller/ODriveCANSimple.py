import can
import enum
import struct
import threading


class AxisState(enum.IntEnum):
    UNDEFINED = 0
    IDLE = 1
    STARTUP_SEQUENCE = 2
    FULL_CALIBRATION_SEQUENCE = 3
    MOTOR_CALIBRATION = 4
    ENCODER_INDEX_SEARCH = 6
    ENCODER_OFFSET_CALIBRATION = 7
    CLOSED_LOOP_CONTROL = 8
    LOCKIN_SPIN = 9
    ENCODER_DIR_FIND = 10
    HOMING = 11
    ENCODER_HALL_POLARITY_CALIBRATION = 12
    ENCODER_HALL_PHASE_CALIBRATION = 13
    ANTICOGGING_CALIBRATION = 14


class ControlMode(enum.IntEnum):
    VOLTAGE_CONTROL = 0
    TORQUE_CONTROL = 1
    VELOCITY_CONTROL = 2
    POSITION_CONTROL = 3


class InputMode(enum.IntEnum):
    INACTIVE = 0
    PASSTHROUGH = 1
    VEL_RAMP = 2
    POS_FILTER = 3
    MIX_CHANNELS = 4
    TRAP_TRAJ = 5
    TORQUE_RAMP = 6
    MIRROR = 7
    TUNING = 8


class ODriveCanSimple:
    def __init__(self, axis_id=0, canbus="can0"):
        self.id = axis_id
        self.bus = can.interface.Bus(channel=canbus, bustype="socketcan")

        # Scaling
        self.gain_turnsPm = 1.0
        self.offset_turns = 0.0
        self.input_vel_scale = 1000
        self.input_torque_scale = 1000

        # Callback registry
        self.callbacks = {}
        self._stop_event = threading.Event()
        self._listener_thread = threading.Thread(
            target=self._listener_loop, daemon=True
        )
        self._listener_thread.start()

    # ----------------- Utility -----------------
    def _send(self, arb_id, payload, rtr=False):
        try:
            msg = can.Message(
                is_extended_id=False,
                arbitration_id=arb_id,
                data=payload,
                is_remote_frame=rtr,
            )
            self.bus.send(msg)
        except Exception as e:
            print(f"[ODriveCanSimple] send error: {e}")

    def close(self):
        """Stop listener thread and close CAN bus."""
        self._stop_event.set()
        self._listener_thread.join(timeout=1.0)
        self.bus.shutdown()

    # ----------------- Listener -----------------
    def _listener_loop(self):
        while not self._stop_event.is_set():
            msg = self.bus.recv(timeout=0.1)
            if not msg:
                continue

            # Which command is this?
            cmd_id = msg.arbitration_id - self.id * 0x20

            # Dispatch callbacks
            if cmd_id == 0x09 and "encoder" in self.callbacks:  # encoder estimates
                try:
                    pos, vel = self.decode_encoder_estimates(msg)
                    self.callbacks["encoder"](pos, vel)
                except Exception as e:
                    print(f"[ODriveCanSimple] decode encoder error: {e}")

            elif cmd_id == 0x14 and "iq" in self.callbacks:  # Iq feedback
                try:
                    iq_set, iq_meas = self.decode_iq(msg)
                    self.callbacks["iq"](iq_set, iq_meas)
                except Exception as e:
                    print(f"[ODriveCanSimple] decode iq error: {e}")

            elif cmd_id == 0x17 and "bus" in self.callbacks:  # bus voltage/current
                try:
                    vbus, ibus = self.decode_bus_measurements(msg)
                    self.callbacks["bus"](vbus, ibus)
                except Exception as e:
                    print(f"[ODriveCanSimple] decode bus error: {e}")

    def on_encoder(self, cb):
        """Register callback: cb(pos, vel)"""
        self.callbacks["encoder"] = cb

    def on_bus(self, cb):
        """Register callback: cb(vbus, ibus)"""
        self.callbacks["bus"] = cb

    def on_iq(self, cb):
        """Register callback: cb(iq_set, iq_meas)"""
        self.callbacks["iq"] = cb

    # ----------------- Scaling -----------------
    def set_calibrated_scale(self, gain_turnsPm, offset_m):
        self.gain_turnsPm = gain_turnsPm
        self.offset_turns = offset_m * gain_turnsPm

    def set_ff_scale(self, input_vel_scale, input_torque_scale):
        self.input_vel_scale = input_vel_scale
        self.input_torque_scale = input_torque_scale

    # ----------------- Commands -----------------
    def set_axis_state(self, state: AxisState):
        cmd_id = 0x07
        arb_id = self.id * 0x20 + cmd_id
        payload = int(state).to_bytes(4, "little")
        self._send(arb_id, payload)

    def set_input_pos(self, pos_m, vel_mPs=0, torque=0):
        pos_turns = pos_m * self.gain_turnsPm + self.offset_turns
        vel_turnsPs = vel_mPs * self.gain_turnsPm
        cmd_id = 0x0C
        arb_id = self.id * 0x20 + cmd_id
        payload = (
            bytearray(struct.pack("<f", pos_turns))
            + int(vel_turnsPs * self.input_vel_scale).to_bytes(
                2, "little", signed=True
            )
            + int(torque * self.input_torque_scale).to_bytes(2, "little", signed=True)
        )
        self._send(arb_id, payload)

    def set_input_vel(self, vel_mPs, torque=0):
        vel_turnsPs = vel_mPs * self.gain_turnsPm
        cmd_id = 0x0D
        arb_id = self.id * 0x20 + cmd_id
        payload = bytearray(struct.pack("<f", vel_turnsPs)) + bytearray(
            struct.pack("<f", torque)
        )
        self._send(arb_id, payload)

    def set_input_torque(self, torque):
        cmd_id = 0x0E
        arb_id = self.id * 0x20 + cmd_id
        payload = bytearray(struct.pack("<f", torque))
        self._send(arb_id, payload)

    def set_controller_mode(self, control_mode: ControlMode, input_mode: InputMode):
        cmd_id = 0x0B
        arb_id = self.id * 0x20 + cmd_id
        payload = int(control_mode).to_bytes(4, "little") + int(input_mode).to_bytes(
            4, "little"
        )
        self._send(arb_id, payload)

    def set_abs_pos(self, pos_turns):
        cmd_id = 0x19
        arb_id = self.id * 0x20 + cmd_id
        payload = bytearray(struct.pack("<f", pos_turns))
        self._send(arb_id, payload)

    # ----------------- Decoders -----------------
    @staticmethod
    def decode_encoder_estimates(msg):
        pos, vel = struct.unpack("<ff", msg.data)
        return pos, vel

    @staticmethod
    def decode_iq(msg):
        iq_set, iq_meas = struct.unpack("<ff", msg.data)
        return iq_set, iq_meas

    @staticmethod
    def decode_bus_measurements(msg):
        vbus, ibus = struct.unpack("<ff", msg.data)
        return vbus, ibus
