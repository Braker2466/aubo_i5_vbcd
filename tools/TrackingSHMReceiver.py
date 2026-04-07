from multiprocessing import shared_memory
import numpy as np
import time

frame_dtype = np.dtype([
    ("valid",      np.uint8),
    ("timestamp",  np.float64),
    ("t_capture",  np.float64),

    ("pixel_x",    np.float32),
    ("pixel_y",    np.float32),

    ("cam_x",      np.float32),
    ("cam_y",      np.float32),
    ("cam_z",      np.float32),
    ("cam_vx",     np.float32),
    ("cam_vy",     np.float32),
])

control_dtype = np.dtype([
    ("write_idx",  np.uint8),   # 0 or 1
    ("seq",        np.uint64),
])


class TrackingSHMReceiver:
    def __init__(self, name="tracking_shm"):
        self.dtype = np.dtype([
            ("timestamp",  np.float64),
            ("t_capture",  np.float64),
            ("pixel_x",    np.float32),
            ("pixel_y",    np.float32),
            ("cam_x",      np.float32),
            ("cam_y",      np.float32),
            ("cam_z",      np.float32),
        ])

        self.shm = shared_memory.SharedMemory(name=name)
        self.buf = np.ndarray((), dtype=self.dtype, buffer=self.shm.buf)

        self.last_timestamp = 0.0

    def recv_latest(self):
        ts = float(self.buf["timestamp"])
        if ts <= self.last_timestamp:
            return None

        self.last_timestamp = ts

        if self.buf["valid"] == 0:
            return {
                "valid": False,
                "timestamp": ts,
                "t_capture": float(self.buf["t_capture"]),
            }

        return {
            "valid": True,
            "timestamp": ts,
            "t_capture": float(self.buf["t_capture"]),
            "pixel": (
                float(self.buf["pixel_x"]),
                float(self.buf["pixel_y"]),
            ),
            "camera": (
                float(self.buf["cam_x"]),
                float(self.buf["cam_y"]),
                float(self.buf["cam_z"]),
            ),
        }

    def close(self):
        self.shm.close()

class TrackingSHMDoubleReceiver:
    def __init__(self, name="tracking_shm_db"):
        self.frame_dtype = frame_dtype
        self.control_dtype = control_dtype

        shm = shared_memory.SharedMemory(name=name)
        buf = shm.buf
        offset = 0

        self.ctrl = np.ndarray((), self.control_dtype, buffer=buf, offset=offset)
        offset += self.control_dtype.itemsize

        self.frames = [
            np.ndarray((), self.frame_dtype, buffer=buf, offset=offset),
            np.ndarray((), self.frame_dtype, buffer=buf, offset=offset + self.frame_dtype.itemsize),
        ]

        self.last_seq = -1
        self.shm = shm

    def recv_latest(self):
        seq = int(self.ctrl["seq"])
        if seq == self.last_seq:
            return None

        self.last_seq = seq
        read_idx = int(self.ctrl["write_idx"])
        frame = self.frames[read_idx]

        if frame["valid"] == 0:
            return {"valid": False}

        return {
            "valid": True,
            "timestamp": float(frame["timestamp"]),
            "t_capture": float(frame["t_capture"]),
            "pixel": (
                float(frame["pixel_x"]),
                float(frame["pixel_y"]),
            ),
            "camera": (
                float(frame["cam_x"]),
                float(frame["cam_y"]),
                float(frame["cam_z"]),
                float(frame["cam_vx"]),
                float(frame["cam_vy"])
            ),
        }

    def close(self):
        self.shm.close()



if __name__ == "__main__":

    receiver = TrackingSHMDoubleReceiver()
    last_print = 0
    try:
        while True:
            data = receiver.recv_latest()
            if data is None:
                continue

            if not data["valid"]:
                # 目标丢失
                continue

            pixel = data["pixel"]
            camera = data["camera"]

            # === 延迟调试 ===
            delay = time.time() - data["t_capture"]
            print(f"[DELAY] {delay*1000:.3f} ms")
    finally:
        receiver.close()   # 只 close，不 unlink
