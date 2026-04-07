import numpy as np
import cv2
import time
from core.Aubo_Robot import Aubo_Robot



# ------------------ 全局变量 ------------------
click_point_pix = None
manual_compensation = np.zeros(3)  # ΔX, ΔY, ΔZ
camera_depth_img = None  # 你的深度图
use_tool = False  # 是否考虑工具对齐



# ------------------ 鼠标回调函数 ------------------
def mouseclick_callback(event, x, y, flags, param):
    global click_point_pix, manual_compensation,last_target_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        # ----------------- 左键点击：粗对齐 -----------------
        click_point_pix = (x, y)
        print(f"pix: {x}, {y}")

        # Get click point in camera coordinates
        click_z = camera_depth_img[y][x] * robot.cam_depth_scale
        click_x = np.multiply(x - robot.cam_intrinsics[0][2], click_z / robot.cam_intrinsics[0][0])
        click_y = np.multiply(y - robot.cam_intrinsics[1][2], click_z / robot.cam_intrinsics[1][1])
        # if click_z == 0:
        #     return
        click_point = np.asarray([click_x, click_y, click_z])
        click_point.shape = (3, 1)

        # Convert camera to robot coordinates
        # camera2robot = np.linalg.inv(robot.cam_pose)
        flange2camera = robot.cam_pose
        current_point=robot.get_current_waypoint()

        base2flange = np.eye(4)
        base2flange[:3, 3] = current_point['pos']
        rpy=robot.quaternion_to_rpy(current_point['ori'])
        base2flange[:3,:3]=robot.rpy2R(rpy)

        base2camera=  base2flange @ flange2camera

        base2obj = np.dot(base2camera[0:3, 0:3], click_point) + base2camera[0:3, 3:]  #执行变换：先旋转，再加平移

        base2obj_position = base2obj[0:3, 0]

        print(base2obj_position)

        # 应用手动补偿
        target_pos = base2obj_position + manual_compensation
        # 粗对齐移动
        robot.align_to_target(target_pos, z_offset=0.3)

        print(f"粗对齐完成，目标基坐标: {target_pos}")
        last_target_pos = target_pos  # 保存粗对齐的目标坐标

    elif event == cv2.EVENT_RBUTTONDOWN:
        # ----------------- 右键点击：记录手动调整 -----------------
        if click_point_pix is None or last_target_pos is None:
            print("请先左键点击目标进行粗对齐")
            return

        current_pose = robot.get_current_waypoint()
        print(f"手动调整末端当前位置: {current_pose['pos']}")

        # 直接使用上次左键点击计算的 target_pos
        delta_xy = np.array(current_pose['pos'][:2]) - last_target_pos[:2]
        manual_compensation[:2] = delta_xy
        manual_compensation[2] = 0.0  # Z不修正

        print(f"记录手动补偿 ΔX, ΔY = {delta_xy}")
        print("下次粗对齐会自动应用此补偿")

cv2.namedWindow("CameraView")
cv2.setMouseCallback("CameraView", mouseclick_callback)
if __name__ == "__main__":
    # Move robot to home pose
    Aubo_Robot.initialize()
    robot = Aubo_Robot(is_use_camera=True,is_use_jaw=False)
    robot.go_home()
    while True:
        camera_color_img, camera_depth_img = robot.get_camera_data()
        cv2.imshow('CameraView', camera_color_img)
        # cv2.imshow('depth', camera_depth_img)

        if cv2.waitKey(1) == ord('c'):
            break

    cv2.destroyAllWindows()
