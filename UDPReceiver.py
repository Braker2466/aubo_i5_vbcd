import socket
import json
import threading
from typing import Optional, Dict
import time

class UDPTrackingReceiver:
    """
    UDP 目标检测/跟踪接收器（后台线程更新最新帧）
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        buffer_size: int = 1024
    ):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.sock.setblocking(False)  # 非阻塞

        # 最新帧数据缓存
        self.latest_payload: Optional[Dict] = None

        # 后台线程控制
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def _receive_loop(self):
        """后台线程不断接收最新数据"""
        while self._running:
            try:
                data, addr = self.sock.recvfrom(self.buffer_size)
                payload = json.loads(data.decode("utf-8"))
                # 更新最新一帧
                self.latest_payload = payload
            except BlockingIOError:
                # 没有数据时直接跳过，非阻塞
                continue
            except Exception as e:
                print(f"[UDPTrackingReceiver] 接收数据出错: {e}")

    def recv_latest(self) -> Optional[Dict]:
        """
        获取最新一帧数据，不阻塞
        返回:
            dict 或 None
        """
        return self.latest_payload

    def close(self):
        """关闭接收器"""
        self._running = False
        self._thread.join(timeout=1.0)
        if self.sock:
            self.sock.close()
            self.sock = None

if __name__ == "__main__":
    receiver = UDPTrackingReceiver(port=9000)
    print("等待接收数据...")
    last_timestamp = None
    try:
        while True:
            data = receiver.recv_latest()
            if data is None:
                continue
            timestamp = data.get("timestamp", None)
            detections = data.get("detections", None)
            prediction = data.get("prediction", None)
            t_capture = data.get("t_capture", None)

            if timestamp is None or prediction is None:
                continue
            if last_timestamp is not None and timestamp <= last_timestamp:
                continue
            last_timestamp = timestamp
            # 直接获取 prediction
            # prediction = data.get("prediction", None)
            # if prediction:
            pixel = prediction.get("pixel", None)
            predict_camera = prediction.get("camera", None)
            # camera = detections[0].get("camera", None)
            # camera = detections.get("camera", None)
            # print(f"Pixel: {pixel}, predict_Camera: {predict_camera}")
            t_now = time.time()
            delay = t_now - t_capture

            print(f"[DELAY] {delay*1000:.1f} ms")

    finally:
        receiver.close()
'''
camera 原始数据: [-8.886024261300918e-06, 6.170539563754573e-05, 0.0003000000142492354]
类型: <class 'numpy.ndarray'>
数据类型 dtype: float64
形状 shape: (3,)


Camera: [0.06920235604047775, 0.026632366701960564, 0.30000001192092896]


原始数据: [[0.06920236]
 [0.0291023 ]
 [0.3       ]]

'''



