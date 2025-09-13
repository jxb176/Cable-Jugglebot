import socket
import json
import threading
import time

# --------------------------
# Config
# --------------------------
CONTROL_PC_IP = "192.168.0.67"   # IP of your control machine
UDP_PORT = 5005
TCP_PORT = 5006

# --------------------------
# UDP Telemetry Sender
# --------------------------
def telemetry_sender():
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        telemetry = {
            "timestamp": time.time(),
            "position": [1.0, 2.0, 3.0],
            "velocity": [0.1, 0.0, 0.2]
        }
        message = json.dumps(telemetry).encode()
        udp_sock.sendto(message, (CONTROL_PC_IP, UDP_PORT))
        time.sleep(0.01)  # 100 Hz


# --------------------------
# TCP Command Receiver (Persistent)
# --------------------------
def command_receiver():
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_sock.bind(("0.0.0.0", TCP_PORT))
    tcp_sock.listen(1)
    print(f"Waiting for command connection on TCP {TCP_PORT}...")

    conn, addr = tcp_sock.accept()
    print("Command connection established from", addr)

    with conn:
        buffer = ""
        while True:
            data = conn.recv(1024)
            if not data:
                print("Command connection closed")
                break
            buffer += data.decode()
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                command = json.loads(line)
                print("Received command:", command)
                # TODO: Handle command here


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    threading.Thread(target=telemetry_sender, daemon=True).start()
    command_receiver()
