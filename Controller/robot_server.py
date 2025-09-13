# robot_server.py
import socket
import json
import time
import threading
import random

TCP_CMD_PORT = 5555
UDP_TELEM_PORT = 5556

class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.controller_ip = None  # set when TCP client connects
        # 6-axis targets (turns) and robot state
        self.axes = [0.0] * 6         # positions in turns
        self.state = "disable"        # "enable" | "disable" | "estop"

    def set_controller_ip(self, ip):
        with self.lock:
            self.controller_ip = ip

    def get_controller_ip(self):
        with self.lock:
            return self.controller_ip

    # Axes and state
    def set_axes(self, positions):
        if not isinstance(positions, (list, tuple)) or len(positions) != 6:
            raise ValueError("positions must be length-6 list/tuple")
        with self.lock:
            self.axes = [float(x) for x in positions]

    def get_axes(self):
        with self.lock:
            return list(self.axes)

    def set_state(self, value: str):
        value = str(value).lower()
        if value not in ("enable", "disable", "estop"):
            raise ValueError("state must be one of: enable, disable, estop")
        with self.lock:
            self.state = value

    def get_state(self) -> str:
        with self.lock:
            return self.state

def tcp_command_server(state: RobotState):
    """Accepts a single TCP client, receives newline-delimited JSON commands."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", TCP_CMD_PORT))
    srv.listen(1)
    print(f"[TCP] Listening on :{TCP_CMD_PORT}")
    while True:
        conn, addr = srv.accept()
        print(f"[TCP] Controller connected from {addr}")
        state.set_controller_ip(addr[0])
        try:
            with conn, conn.makefile("r") as f:
                for line in f:
                    try:
                        msg = json.loads(line.strip())
                        mtype = msg.get("type")
                        if mtype == "axes":
                            positions = msg.get("positions", [])
                            # units = msg.get("units", "turns")
                            state.set_axes(positions)
                        elif mtype == "state":
                            state.set_state(msg.get("value", "disable"))
                        else:
                            print("[TCP] Unknown command type:", mtype)
                    except Exception as e:
                        print("[TCP] Bad command:", e)
        except Exception as e:
            print("[TCP] Connection error:", e)
        finally:
            print("[TCP] Controller disconnected")

def udp_telemetry_sender(state: RobotState):
    """Sends telemetry to the controller IP over UDP at 10 Hz."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()
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
            print("[UDP] Telemetry send error:", e)

def axes_state_logger(state: RobotState):
    """Prints current robot state and 6-axis targets at 1 Hz."""
    while True:
        try:
            axes = state.get_axes()
            st = state.get_state()
            print(
                f"[LOG] State={st} Axes(turns)=["
                + ", ".join(f"{x:.3f}" for x in axes)
                + "]"
            )
        except Exception as e:
            print(f"[LOG] Error reading state/axes: {e}")
        time.sleep(1.0)

if __name__ == "__main__":
    state = RobotState()
    threading.Thread(target=tcp_command_server, args=(state,), daemon=True).start()
    threading.Thread(target=udp_telemetry_sender, args=(state,), daemon=True).start()
    threading.Thread(target=axes_state_logger, args=(state,), daemon=True).start()
    print("Robot server running. Press Ctrl+C to exit.")
    while True:
        time.sleep(1)