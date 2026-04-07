from multiprocessing import Process, Queue
import numpy as np
import time
import cv2
from core.Aubo_Robot import Aubo_Robot
from UDPReceiver import UDPTrackingReceiver
from TrackingSHMReceiver import TrackingSHMDoubleReceiver
from core.Camera import RSD435i

def put_latest(mp_queue, data):
    while True:
        try:
            mp_queue.put_nowait(data)
            break
        except:
            try:
                mp_queue.get_nowait()  # 丢掉旧数据
            except:
                break


def UDP_receiver_process(queue):
    receiver = UDPTrackingReceiver(port=9000)
    last_timestamp = None
    print("Receiver process started, waiting for data...")
    while True:
        data = receiver.recv_latest()
        if data is None:
            continue

        ts = data.get("timestamp")
        if ts is None:
            continue

        if last_timestamp is not None and ts <= last_timestamp:
            continue

        last_timestamp = ts

        put_latest(queue, data)

def SHM_receiver_process(queue):
    """
    接收 SHM 双缓冲最新数据，并放入控制进程队列
    """
    receiver = TrackingSHMDoubleReceiver()
    print("[SHM Receiver] Process started, waiting for data...")
    # last_seq = -1  # 用于丢弃重复帧
    try:
        while True:
            data = receiver.recv_latest()
            if data is None or not data.get("valid", False):
                # 没有有效数据就跳过
                time.sleep(0.001)  # CPU-friendly
                continue

            # seq = data.get("seq", None)
            # if seq is None or seq == last_seq:
            #     # 重复帧，不处理
            #     continue

            # last_seq = seq

            # 放入控制进程队列（非阻塞，保证最新帧）
            try:
                put_latest(queue, data)
            except:
                # 队列满就丢掉，保持最新
                pass
    except KeyboardInterrupt:
        print("Receiver exiting...")
    finally:
        receiver.close()   # 只 close，不 unlink

def control_process(queue):
    Aubo_Robot.initialize()
    robot = Aubo_Robot()
    robot.go_home()
    robot.set_param()
    last_data = None
    data = None
    camera = None
    center = None

    # CONTROL_PERIOD = 0.05  # 20 Hz
    while True:

        try:
            # 非阻塞取最新
            data = queue.get_nowait()
        except:
            pass

        if data is not None:
        #     prediction = data.get("prediction",None)
        #     detections = data.get("detections", None)
        #     if prediction is not None:
        #         camera_info = prediction.get("camera", None)
        #         camera = camera_info[:3] if camera_info is not None else None
        #         vel = camera_info[3:] if camera_info is not None else None
        #     if detections is not None:
        #         center = detections[0].get("center", None)#这里是默认显示第一个检测到的目标，可以根据需要修改
            # # else:
            # #     camera = predict_camera

            camera = data.get("camera", None)
            center = data.get("pixel", None)

            if camera: 
                target_pos = robot.compute_target2base(camera[:3])
                print("detect_center:", center)
                print("vel:", camera[3:])
                # print("vel:", vel)
                robot.align_to_target_line(target_pos, z_offset=0.5,use_fixed_z=True)
                data = None


if __name__ == "__main__":
    queue = Queue(maxsize=1)

    # p_recv = Process(target=UDP_receiver_process, args=(queue,))
    p_recv = Process(target=SHM_receiver_process, args=(queue,))
    p_ctrl = Process(target=control_process, args=(queue,))

    p_recv.start()
    p_ctrl.start()

    p_recv.join()
    p_ctrl.join()
