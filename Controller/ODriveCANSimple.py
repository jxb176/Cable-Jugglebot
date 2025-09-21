# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:35:04 2024

@author: jonbe
"""

import can
import enum
import struct


class ODriveCanSimple:
    def __init__(self, axis_id=0, canbus='can0'):
        self.id = axis_id
        self.bus = canbus
        self.gain_turnsPm = 1.0
        self.offset_turns = 0.0
        self.input_vel_scale = 1000
        self.input_torque_scale = 1000

    def set_calibrated_scale(self, gain_turnsPm, offset_m):
        self.gain_turnsPm = gain_turnsPm
        self.offset_turns = offset_m * gain_turnsPm

    def set_ff_scale(self, input_vel_scale,
                     input_torque_scale):  # Send the commands to the ODrive to set this in the future
        self.input_vel_scale = input_vel_scale
        self.input_torque_scale = input_torque_scale

    def set_axis_state(self, state):
        cmd_id = 0x07
        arb_id = self.id * 0x20 + cmd_id
        payload = int(state).to_bytes(4, 'little')
        msg = can.Message(is_extended_id=False, arbitration_id=arb_id, data=payload)
        self.bus.send(msg)

    def set_input_pos(self, pos_m, vel_mPs=0, torque=0):
        pos_turns = pos_m * self.gain_turnsPm + self.offset_turns
        vel_turnsPs = vel_mPs * self.gain_turnsPm

        cmd_id = 0x0c
        arb_id = self.id * 0x20 + cmd_id
        payload = bytearray(struct.pack("f", pos_turns)) + int(vel_turnsPs * self.input_vel_scale).to_bytes(2, 'little',
                                                                                                            signed=True) + int(
            torque * self.input_torque_scale).to_bytes(2, 'little', signed=True)
        msg = can.Message(is_extended_id=False, arbitration_id=arb_id, data=payload)
        self.bus.send(msg)

    def set_input_vel(self, vel_mPs, torque=0):
        vel_turnsPs = vel_mPs * self.gain_turnsPm

        cmd_id = 0x0d
        arb_id = self.id * 0x20 + cmd_id
        payload = bytearray(struct.pack("f", vel_turnsPs)) + bytearray(struct.pack("f", torque))
        msg = can.Message(is_extended_id=False, arbitration_id=arb_id, data=payload)
        self.bus.send(msg)

    def set_input_torque(self, torque):
        cmd_id = 0x0e
        arb_id = self.id * 0x20 + cmd_id
        payload = bytearray(struct.pack("f", torque))
        msg = can.Message(is_extended_id=False, arbitration_id=arb_id, data=payload)
        self.bus.send(msg)

    # test me

    def set_controller_mode(self, control_mode, input_mode):
        cmd_id = 0x0b
        arb_id = self.id * 0x20 + cmd_id
        payload = int(control_mode).to_bytes(4, 'little') + int(input_mode).to_bytes(4, 'little')
        msg = can.Message(is_extended_id=False, arbitration_id=arb_id, data=payload)
        self.bus.send(msg)

    def set_abs_pos(self, pos):
        cmd_id = 0x19
        arb_id = self.id * 0x20 + cmd_id
        payload = bytearray(struct.pack("f", pos))
        msg = can.Message(is_extended_id=False, arbitration_id=arb_id, data=payload)
        self.bus.send(msg)

    def request_encoder_estimates(self):
        cmd_id = 0x09
        arb_id = self.id * 0x20 + cmd_id
        payload = bytearray([])
        msg = can.Message(is_extended_id=False, is_remote_frame=True, arbitration_id=arb_id, data=payload)
        self.bus.send(msg)

    def request_iq(self):
        cmd_id = 0x14
        arb_id = self.id * 0x20 + cmd_id
        payload = bytearray([])
        msg = can.Message(is_extended_id=False, is_remote_frame=True, arbitration_id=arb_id, data=payload)
        self.bus.send(msg)

    # implement
    '''
    def get_error(self):
        pass        


    def set_limits(self):
        pass
    def clear_errors(self):
        pass
    def set_pos_gain(self):
        pass
    def set_vel_gains(self):
        pass
    def get_powers(self):
        pass
    Set position command limits? - this probably belongs in the motion controller, but might make sense to store values in the odrv/actuator library?
    '''


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

