# udp_sniff.py
import socket, json
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 5556))
print("Listening on UDP :5556 ...")
while True:
    data, addr = s.recvfrom(65535)
    print("FROM", addr, "RAW:", data)
    try:
        j = json.loads(data.decode("utf-8", errors="ignore"))
        print("JSON keys:", list(j.keys()), "sample:", j)
    except Exception as e:
        print("Decode error:", e)