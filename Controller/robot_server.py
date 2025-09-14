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
        # Profile storage and player
        self.profile = []             # list of tuples: (t, [6 positions])
        self.player_thread = None     # type: ProfilePlayer | None

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

    # Profile management
    def set_profile(self, profile_points):
        """profile_points: list of [t, a1..a6]"""
        if not isinstance(profile_points, (list, tuple)) or len(profile_points) == 0:
            raise ValueError("profile must be a non-empty list")
        prof = []
        for row in profile_points:
            if not isinstance(row, (list, tuple)) or len(row) < 7:
                raise ValueError("each profile row must be [t, a1..a6]")
            t = float(row[0])
            axes = [float(x) for x in row[1:7]]
            prof.append((t, axes))
        # Ensure non-decreasing time
        times = [p[0] for p in prof]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("profile time column must be non-decreasing")
        with self.lock:
            self.profile = prof

    def get_profile(self):
        with self.lock:
            return list(self.profile)

    def start_profile(self, rate_hz: float):
        self.stop_profile()
        prof = self.get_profile()
        if not prof:
            raise RuntimeError("no profile uploaded")
        player = ProfilePlayer(self, prof, rate_hz)
        with self.lock:
            self.player_thread = player
        player.start()

    def stop_profile(self):
        with self.lock:
            player = self.player_thread
            self.player_thread = None
        if player and player.is_alive():
            player.stop()
            player.join(timeout=1.0)


class ProfilePlayer(threading.Thread):
    """Plays a time-position profile with linear interpolation at fixed rate."""
    def __init__(self, state: RobotState, profile, rate_hz: float):
        super().__init__(daemon=True)
        self.state = state
        self.profile = profile  # list[(t, [6])]
        self.dt = 1.0 / max(1e-3, float(rate_hz))
        self._stop = threading.Event()

        # Pre-normalize time to start at zero
        t0 = self.profile[0][0]
        self.norm_profile = [(t - t0, axes) for (t, axes) in self.profile]
        self.duration = self.norm_profile[-1][0]

    def stop(self):
        self._stop.set()

    def run(self):
        print(f"[PROFILE] Starting playback at {1.0/self.dt:.1f} Hz, duration {self.duration:.3f}s")
        start = time.perf_counter()
        k = 0  # segment index: between norm_profile[k] and [k+1]
        while not self._stop.is_set():
            t = time.perf_counter() - start
            if t >= self.duration:
                # Clamp to final point and finish
                self.state.set_axes(self.norm_profile[-1][1])
                print("[PROFILE] Completed")
                break

            # Advance segment to bracket current time
            while k + 1 < len(self.norm_profile) and self.norm_profile[k + 1][0] < t:
                k += 1
            # Find the two points for interpolation
            t0, p0 = self.norm_profile[k]
            t1, p1 = self.norm_profile[min(k + 1, len(self.norm_profile) - 1)]
            if t1 <= t0:
                alpha = 0.0
            else:
                alpha = (t - t0) / (t1 - t0)
                if alpha < 0.0:
                    alpha = 0.0
                elif alpha > 1.0:
                    alpha = 1.0
            # Linear interpolate each axis
            axes = [p0[i] + alpha * (p1[i] - p0[i]) for i in range(6)]
            self.state.set_axes(axes)

            # Sleep to next tick
            time.sleep(self.dt)
        # Clear active player reference if we are the current one
        with self.state.lock:
            if self.state.player_thread is self:
                self.state.player_thread = None


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
                            state.set_axes(msg.get("positions", []))
                        elif mtype == "state":
                            state.set_state(msg.get("value", "disable"))
                        elif mtype == "profile_upload":
                            profile = msg.get("profile", [])
                            state.set_profile(profile)
                            print(f"[PROFILE] Uploaded {len(state.get_profile())} points")
                        elif mtype == "profile_start":
                            rate_hz = float(msg.get("rate_hz", 100.0))
                            state.start_profile(rate_hz)
                        elif mtype == "profile_stop":
                            state.stop_profile()
                            print("[PROFILE] Stopped")
                        else:
                            print("[TCP] Unknown command type:", mtype)
                    except Exception as e:
                        print("[TCP] Bad command:", e)
        except Exception as e:
            print("[TCP] Connection error:", e)
        finally:
            state.stop_profile()
            print("[TCP] Controller disconnected")

def udp_telemetry_sender(state: RobotState):
    """Sends telemetry to the controller IP over UDP at 10 Hz."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
            print(f"[LOG] State={st} Axes(turns)=[" +
                  ", ".join(f"{x:.3f}" for x in axes) + "]")
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