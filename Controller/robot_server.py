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
        self.speed = 0.0
        self.lock = threading.Lock()
        self.controller_ip = None  # set when TCP client connects

    def set_speed(self, v):
        with self.lock:
            self.speed = float(v)

    def get_speed(self):
        with self.lock:
            return self.speed

    def set_controller_ip(self, ip):
        with self.lock:
            self.controller_ip = ip

    def get_controller_ip(self):
        with self.lock:
            return self.controller_ip

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
                        if msg.get("type") == "speed":
                            state.set_speed(msg["value"])
                            # optional: acknowledge
                            # conn.sendall(b'{"ok":true}\n')
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
        # Generate telemetry as example: sensor value influenced by speed
        speed = state.get_speed()
        val = random.random() * 2 + 0.1 * speed
        msg = {"t": time.time(), "val": float(val)}
        try:
            sock.sendto(json.dumps(msg).encode("utf-8"), (ctrl_ip, UDP_TELEM_PORT))
        except Exception as e:
            print("[UDP] Telemetry send error:", e)

if __name__ == "__main__":
    state = RobotState()
    threading.Thread(target=tcp_command_server, args=(state,), daemon=True).start()
    threading.Thread(target=udp_telemetry_sender, args=(state,), daemon=True).start()
    print("Robot server running. Press Ctrl+C to exit.")
    while True:
        time.sleep(1)