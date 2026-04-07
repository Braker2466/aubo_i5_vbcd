"""采集相机的照片和机械臂的位姿并保存成文件。
这里以intel realsense 相机为例 """

import cv2
import numpy as np
import pyrealsense2 as rs
from core.Camera import RSD435i
from core.Aubo_Robot import Aubo_Robot

count = 0


image_save_path = "./collect_data5/"


def data_collect():
    global count
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                           cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("detection", color_image)  # 窗口显示，显示名为 Capture_Video

        k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
        if k == ord('s'):  # 键盘按一下s, 保存当前照片和机械臂位姿
            print(f"采集第{count}组数据...")
            # 获取当前位置
            current_pos = robot.get_current_waypoint()

            current_pos_rpy = robot.quaternion_to_rpy(current_pos['ori'])  # 四元数转欧拉角

            pose = current_pos['pos']+current_pos_rpy  # 获取当前机械臂状态 需要根据实际使用的机械臂获得
            print(f"机械臂pose:{pose}")

            with open(f'{image_save_path}poses.txt', 'a+') as f:
                # 将列表中的元素用空格连接成一行
                pose_ = [str(i) for i in pose]
                new_line = f'{",".join(pose_)}\n'
                # 将新行附加到文件的末尾
                f.write(new_line)

            cv2.imwrite(image_save_path + 'images' +str(count) + '.jpg', color_image)
            count += 1

def data_collect2():
    global count
    while True:
        color_image, depth_image = camera.get_data()

        if color_image is None:
            continue

        cv2.imshow("detection", color_image)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            print(f"采集第 {count+1} 组数据...")

            current_pos = robot.get_current_waypoint()
            rpy = robot.quaternion_to_rpy(current_pos['ori'])
            pose = current_pos['pos'] + rpy

            with open(f'{image_save_path}poses.txt', 'a+') as f:
                f.write(','.join(map(str, pose)) + '\n')

            cv2.imwrite(
                image_save_path + f'images{count}.jpg',
                color_image
            )

            # 如果你之后要用深度做标定，强烈建议一起存
            np.save(
                image_save_path + f'depth{count}.npy',
                depth_image
            )

            count += 1


if __name__ == "__main__":
    Aubo_Robot.initialize()

    robot = Aubo_Robot(is_use_jaw=False)
    camera = RSD435i(width=1280,height=720,fps=30)

    robot.go_home()

    data_collect2()
