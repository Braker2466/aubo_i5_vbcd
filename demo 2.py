#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
# from realsenseD415 import Camera
from core.Aubo_Robot import Aubo_Robot
from core.Camera import RSD435i

# Move robot to home pose
Aubo_Robot.initialize()
robot = Aubo_Robot(robot_host_ip="192.168.1.40")
camera = RSD435i(width=1280,height=720,fps=30)  # 深度相机
robot.go_home()


# Callback function for clicking on OpenCV window
click_point_pix = ()
camera_color_img, camera_depth_img = camera.get_data()


def mouseclick_callback(event, x, y, flags, param):
    use_tool=False
    if event == cv2.EVENT_LBUTTONDOWN:
        global camera, robot, click_point_pix
        # x, y = int(robot.cam_intrinsics[0][2]), int(robot.cam_intrinsics[1][2])
        click_point_pix = (x, y)
        print(f"pix: {x}, {y}")
        # Get click point in camera coordinates
        click_z = camera_depth_img[y][x] * robot.cam_depth_scale
        click_x = np.multiply(x - camera.cam_intrinsics[0][2], click_z / camera.cam_intrinsics[0][0])
        click_y = np.multiply(y - camera.cam_intrinsics[1][2], click_z / camera.cam_intrinsics[1][1])
        if click_z == 0:
            return
        click_point = np.asarray([click_x, click_y, click_z])
        click_point.shape = (3, 1)

        print("click_point in camera coordinates:", click_point.flatten())

        flange2camera = robot.cam_pose
        current_point=robot.get_current_waypoint()

        base2flange = np.eye(4)
        base2flange[:3, 3] = current_point['pos']
        rpy=robot.quaternion_to_rpy(current_point['ori'])
        base2flange[:3,:3]=robot.rpy2R(rpy)
        print("base2flange:", base2flange[:3, 3].flatten())
        base2camera=  base2flange @ flange2camera
        print("base2camera:", base2camera[:3, 3].flatten())
        base2obj = np.dot(base2camera[0:3, 0:3], click_point) + base2camera[0:3, 3:]  #执行变换：先旋转，再加平移

        base2obj_position = base2obj[0:3, 0] #这里得到的是目标相对于机器人基座的坐标

        print("base2obj_position:",base2obj_position)
        # print(base2obj_position.shape)
        if use_tool:
            flange2tool = robot.tool_pose
        else :
            flange2tool = flange2camera

        base2target = np.eye(4)
        # base2obj[:3, :3] = robot.rpy2R([(180 / 360.0) * 2 * np.pi, 0, (90 / 360.0) * 2 * np.pi]) #人为指定了目标物体的姿态
        base2target[:3, :3] = base2camera[:3, :3]
        base2target[:3,3]=base2obj_position

        base2flange_expect = base2target @ np.linalg.inv(flange2tool)

        target_pos = base2flange_expect[:3,3]
        target_pos = base2flange_expect[:3,3]
        # target_pos=base2obj_position #法兰对准
        print("target_pos:", target_pos)

        # robot.align_to_target(target_pos, z_offset=0.3)

if __name__ == "__main__":
    # Show color and depth frames
    cv2.namedWindow('color',cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('color', 640, 480)          # 强制显示尺寸 = 图像尺寸
    # cv2.imshow('color', camera_color_img)
    cv2.setMouseCallback('color', mouseclick_callback, camera_color_img)
    cv2.namedWindow('depth')

    while True:
        camera_color_img, camera_depth_img = camera.get_data()
        # bgr_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
        # if len(click_point_pix) != 0:
        #     camera_color_img = cv2.circle(camera_color_img, click_point_pix, 7, (0, 0, 255), 2)
        cx, cy = int(camera.intrinsics[0][2]), int(camera.intrinsics[1][2])
        cv2.circle(camera_color_img, (cx, cy), 5, (0,255,0), -1)  # 绿色 = 相机光学中心
        cv2.imshow('color', camera_color_img)
        cv2.imshow('depth', camera_depth_img)

        if cv2.waitKey(1) == ord('c'):
            break

    cv2.destroyAllWindows()



"""
用自己标的内参
pix: 321, 250
click_point in camera coordinates: [-2.5241339e-04 -2.4818041e-04  3.5300002e-01]
 base2flange: [-0.12149997  0.65092841  0.37427504]
 base2camera: [-0.2088661   0.6176671   0.34791934]
 base2obj_position: [-0.20927519  0.61465094 -0.00506773]
 target_pos: [-0.12190906  0.64791225  0.02128797]
 
pix: 319, 250
click_point in camera coordinates: [-5.6963583e-04 -5.0466206e-05  3.5300002e-01]
base2flange: [-0.12149997  0.65092841  0.37427504]
base2camera: [-0.2088661   0.6176671   0.34791934]
base2obj_position: [-0.20907185  0.61433729 -0.00506537]
target_pos: [-0.12170572  0.64759861  0.02129033]
 
 
 
 
 """



