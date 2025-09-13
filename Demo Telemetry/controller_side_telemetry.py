import socket
import json
import threading
import time

PI_IP = "192.168.1.50"   # Raspberry Pi IP
UDP_PORT = 5005
TCP_PORT = 5006

# --------------------------
# UDP Telemetry Listener
# --------------------------
def telemetry_listener():
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind(("0.0.0.0", UDP_PORT))
    while True:
        data, _ = udp_sock.recvfrom(4096)
        telemetry = json.loads(data.decode())
        print("Telemetry:", telemetry)


# --------------------------
# TCP Command Client (Persistent)
# --------------------------
class CommandClient:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
    
    def send_command(self, command: dict):
        message = json.dumps(command) + "\n"  # newline-delimited messages
        self.sock.sendall(message.encode())
    
    def close(self):
        self.sock.close()


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # Start telemetry listener in background
    threading.Thread(target=telemetry_listener, daemon=True).start()

    # Open persistent TCP connection
    client = CommandClient(PI_IP, TCP_PORT)

    # Example usage
    time.sleep(1)
    client.send_command({"action": "move", "target": [5.0, 2.0, 1.0]})
    time.sleep(1)
    client.send_command({"action": "stop"})
    time.sleep(1)
    client.send_command({"action": "set_speed", "value": 0.5})

    # Keep program alive to receive telemetry
    while True:
        time.sleep(1)
