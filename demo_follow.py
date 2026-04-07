#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
# from realsenseD415 import Camera
from core.Aubo_Robot import Aubo_Robot
from tools.UDPReceiver import UDPTrackingReceiver
from core.Camera import RSD435i

def click2target(robot,camera):

    def mouseclick_callback(event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # x, y = int(robot.cam_intrinsics[0][2]), int(robot.cam_intrinsics[1][2])
            print(f"pix: {x}, {y}")
            # Get click point in camera coordinates
            click_z = camera_depth_img[y][x] * camera.scale
            print("camera.scale:", camera.scale)
            if not np.isfinite(click_z) or click_z <= 0.1:
                print("无效的深度值/深度低于0.1m，强制置0.3m")
                click_z = 0.3 #这里强制把深度设为0.3m，避免除0错误
                # return

            click_x = np.multiply(x - camera.intrinsics[0][2], click_z / camera.intrinsics[0][0])
            click_y = np.multiply(y - camera.intrinsics[1][2], click_z / camera.intrinsics[1][1])
            click_point = np.asarray([click_x, click_y, click_z])
            click_point.shape = (3, 1)

            print("原始数据:", click_point)

            target_pos = robot.compute_target2base(click_point)
            print("target_pos:", target_pos)

            robot.align_to_target(target_pos, z_offset=0.4)

    # Callback function for clicking on OpenCV window
    camera_color_img, camera_depth_img = camera.get_data()
    cv2.namedWindow('color',cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('color', mouseclick_callback, camera_color_img)
    while True:
        camera_color_img, camera_depth_img = camera.get_data()

        cx, cy = int(camera.intrinsics[0][2]), int(camera.intrinsics[1][2])
        cv2.circle(camera_color_img, (cx, cy), 2, (0,255,0), -1)  # 绿色 = 相机光学中心
        cv2.imshow('color', camera_color_img)

        if cv2.waitKey(1) == ord('c'):
            break

    cv2.destroyAllWindows()

def follow_target():

    receiver = UDPTrackingReceiver(port=9000)
    print("等待接收数据...")
    last_timestamp = None

    try:
        while True:
            data = receiver.recv_latest()
            if data is None:
                continue

            timestamp = data.get("timestamp", None)
            prediction = data.get("prediction", None)
            detections = data.get("detections", None)
            t_capture = data.get("t_capture", None)

            if timestamp is None or prediction is None:
                continue

            # 如果不是新数据，直接跳过
            if last_timestamp is not None and timestamp <= last_timestamp:
                continue
            
            last_timestamp = timestamp  # 只在新数据时更新
            # predict_camera = prediction.get("camera", None)
            if detections is not None:
                camera = detections[0].get("camera", None)
                center = detections[0].get("center", None)
            else :
                camera = prediction.get("camera", None)
            
            if camera: 
                target_pos = robot.compute_target2base(camera)
                print("center:", center)
                # t_now = time.time()
                # delay = t_now - t_capture

                # print(f"[DELAY] {delay*1000:.1f} ms")
                # robot.align_to_target_line(target_pos, z_offset=0.3)
    finally:
        receiver.close()


if __name__ == "__main__":
    Aubo_Robot.initialize()
    # camera = RSD435i(width=1280,height=720,fps=30)  # 深度相机
    robot = Aubo_Robot(robot_host_ip="192.168.1.40")
    robot.set_end_max_line_velc(0.02)
    robot.set_end_max_line_acc(0.02)
    # # 设置关节最大加速度
    # robot.set_joint_maxacc((1, 1, 1, 1, 1, 1))
    # # 设置关节最大加速度
    # robot.set_joint_maxvelc((1, 1, 1, 1, 1, 1))
    robot.get_info()
    robot.go_home()
    # click2target(robot,camera)
    # follow_target()


   



"""

在这个程序里初步调试了通过udp接受坐标并移动机械臂跟踪的功能，
首先是发现闭环跟踪的时候震荡很厉害，中间考虑了kalman的预测是不是发力了，但是
调试发现这部分影响甚微，问题来自处理的图片不是当前或者说附近的帧，
发现是延迟过大，原因是内部的机械臂控制器通信上耗时了，这里阻塞等待了，
所以在正式的run.py里用多进程的方式将接收端和机械臂运动独立开，显著减少了延迟。
 
 """


