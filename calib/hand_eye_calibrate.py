# coding=utf-8
"""
眼在手上 用采集到的图片信息和机械臂位姿信息计算 相机坐标系相对于机械臂末端坐标系的 旋转矩阵和平移向量
A2^{-1}*A1*X=X*B2*B1^{−1}
"""

import os

import cv2
import numpy as np
from calib.calibration_store import LEGACY_PATHS, update_system_calibration

np.set_printoptions(precision=8, suppress=True)

iamges_path = "./collect_data"  # 手眼标定采集的标定版图片所在路径
arm_pose_file = "./collect_data/poses.txt"  # 采集标定板图片时对应的机械臂末端的位姿 从 第一行到最后一行 需要和采集的标定板的图片顺序进行对应
N=24  # 采集图像数量

def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    return R, t


def camera_calibrate(N,iamges_path):
    print("++++++++++开始相机标定++++++++++++++")
    # 角点的个数以及棋盘格间距
    XX = 11  # 标定板的中长度对应的角点的个数
    YY = 8  # 标定板的中宽度对应的角点的个数
    L = 0.015  # 标定板一格的长度  单位为米

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
    objp = L * objp

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    for i in range(0, N):  # 标定好的图片在iamges_path路径下，从0.jpg到x.jpg   一般采集20张左右就够，实际情况可修改

        image = f"{iamges_path}/images{i}.jpg"
        print(f"正在处理第{i}张图片：{image}")

        if os.path.exists(image):

            img = cv2.imread(image)
            print(f"图像大小： {img.shape}")
            # h_init, width_init = img.shape[:2]
            # img = cv2.resize(src=img, dsize=(width_init // 2, h_init // 2))
            # print(f"图像大小(resize)： {img.shape}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
            # print(corners)
            print(f"左上角点：{corners[0, 0]}")
            print(f"右下角点：{corners[-1, -1]}")

            # 绘制角点并显示图像
            cv2.drawChessboardCorners(img, (XX, YY), corners, ret)
            cv2.imshow('Chessboard', img)

            cv2.waitKey(1000)  ## 停留1s, 观察找到的角点是否正确

            if ret:

                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

    N = len(img_points)

    # 标定得到图案在相机坐标系下的位姿
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    # print("ret:", ret)
    print("内参矩阵:\n", mtx)  # 内参数矩阵
    print("畸变系数:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)

    print("++++++++++相机标定完成++++++++++++++")

    return rvecs, tvecs


def camera_calibrate_2(N, iamges_path):
    print("++++++++++开始相机标定++++++++++++++")

    # 角点的个数以及棋盘格间距
    XX = 11  # 标定板长度方向内角点个数
    YY = 8   # 标定板宽度方向内角点个数
    L = 0.015  # 棋盘格单格边长，单位 m

    # 保存角点可视化图片的目录
    save_vis_dir = os.path.join(iamges_path, "corner_vis")
    os.makedirs(save_vis_dir, exist_ok=True)

    # 亚像素角点停止准则
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 构造棋盘格世界坐标
    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2)
    objp = L * objp

    obj_points = []  # 3D点
    img_points = []  # 2D点

    size = None
    valid_image_indices = []

    for i in range(0, N):
        image = f"{iamges_path}/images{i}.jpg"
        print(f"\n正在处理第{i}张图片：{image}")

        if not os.path.exists(image):
            print("图片不存在，跳过")
            continue

        img = cv2.imread(image)
        if img is None:
            print("图片读取失败，跳过")
            continue

        print(f"图像大小：{img.shape}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, (XX, YY), None
        )

        vis = img.copy()

        if ret:
            # 亚像素优化
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            print(f"左上角点：{corners2[0, 0]}")
            print(f"右下角点：{corners2[-1, -1]}")

            # 绘制角点
            cv2.drawChessboardCorners(vis, (XX, YY), corners2, ret)

            # 加入标定数据
            obj_points.append(objp)
            img_points.append(corners2)
            valid_image_indices.append(i)

            status_text = "Corners: OK"
            status_color = (0, 255, 0)

        else:
            print("未检测到棋盘格角点")
            status_text = "Corners: FAIL"
            status_color = (0, 0, 255)

        # 叠加显示信息
        # cv2.putText(vis, f"Image: images{i}.jpg", (20, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # cv2.putText(vis, status_text, (20, 65),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        # cv2.putText(vis, "Enter: next | s: save | Esc: quit", (20, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 自适应缩放显示
        h, w = vis.shape[:2]
        scale = min(1400 / w, 900 / h, 1.0)
        show_img = cv2.resize(vis, (int(w * scale), int(h * scale)))

        cv2.imshow("Chessboard", show_img)

        while True:
            key = cv2.waitKey(0) & 0xFF

            # Enter -> 下一张
            if key == 13:
                save_name = f"images{i}_corners.png"
                save_path = os.path.join(save_vis_dir, save_name)
                cv2.imwrite(save_path, vis)
                print(f"已保存角点可视化图：{save_path}")
                print("进入下一张")
                break


            # Esc -> 提前退出
            elif key == 27:
                print("用户终止相机标定图片浏览")
                cv2.destroyAllWindows()

                if len(img_points) < 3:
                    raise ValueError("有效标定图片数量不足，至少需要3张以上成功检测角点的图片")

                ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj_points, img_points, size, None, None
                )

                print("内参矩阵:\n", mtx)
                print("畸变系数:\n", dist)
                print("有效图像编号:", valid_image_indices)
                print("++++++++++相机标定完成++++++++++++++")
                return rvecs, tvecs

    cv2.destroyAllWindows()

    if len(img_points) < 3:
        raise ValueError("有效标定图片数量不足，至少需要3张以上成功检测角点的图片")

    # 相机标定
    ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, size, None, None
    )

    print("内参矩阵:\n", mtx)
    print("畸变系数:\n", dist)
    print("有效图像编号:", valid_image_indices)
    print("++++++++++相机标定完成++++++++++++++")

    return rvecs, tvecs


def process_arm_pose(arm_pose_file):
    """处理机械臂的pose文件。 采集数据时， 每行保存一个机械臂的pose信息， 该pose与拍摄的图片是对应的。
    pose信息用6个数标识， 【x,y,z,Rx, Ry, Rz】. 需要把这个pose信息用旋转矩阵表示。rx, ry, rz为弧度值"""

    R_arm, t_arm = [], []
    with open(arm_pose_file, "r", encoding="utf-8") as f:
        # 读取文件中的所有行
        all_lines = f.readlines()
    for line in all_lines:
        pose = [float(v) for v in line.split(',')]
        R, t = pose_to_homogeneous_matrix(pose=pose)
        R_arm.append(R)
        t_arm.append(t)
    return R_arm, t_arm


def hand_eye_calibrate():
    rvecs, tvecs = camera_calibrate_2(N,iamges_path=iamges_path)
    #_2会保存图片，1不会
    R_arm, t_arm = process_arm_pose(arm_pose_file=arm_pose_file)

    R, t = cv2.calibrateHandEye(R_arm, t_arm, rvecs, tvecs, cv2.CALIB_HAND_EYE_TSAI)
    print("+++++++++++手眼标定完成+++++++++++++++")
    return R, t


def save_hand_eye_result(R, t):
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = R
    camera_pose[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    np.savetxt(LEGACY_PATHS["camera_pose"], camera_pose, fmt="%.8f")
    update_system_calibration(camera_pose=camera_pose)
    print(f"手眼标定结果已保存到 {LEGACY_PATHS['camera_pose']}")
    print("手眼标定结果已同步写入 calib/config/system_calibration.json")
    return camera_pose


if __name__ == "__main__":
    R, t = hand_eye_calibrate()
    save_hand_eye_result(R, t)

    print("旋转矩阵：")
    print(R)
    print("平移向量：")
    print(t)

'''


cam_intrinsics = np.array([607.005, 0, 319.980, 0, 607.304, 250.081, 0, 0, 1])

++++++++++相机标定完成++++++++++++++
内参矩阵:
 [[604.93827696   0.         321.24121602]
 [  0.         605.34683728 250.65016923]
 [  0.           0.           1.        ]]
畸变系数:
 [[ 0.01161291  0.8543101   0.0001343   0.00150986 -3.20276531]]
 
+++++++++++手眼标定完成+++++++++++++++
旋转矩阵：
[[-0.69321371 -0.72072316  0.0035881 ]
 [ 0.72073179 -0.69320022  0.00437643]
 [-0.00066693  0.00561986  0.99998399]]
平移向量：
[[0.08481401]
 [0.03533026]
 [0.02578242]]


出厂标定：[ 640x480  p[319.98 250.087]  f[607.005 607.304]  Inverse Brown Conrady [0 0 0 0 0] ]

 内参矩阵:
 [[602.50928111   0.         321.26031868]
 [  0.         602.83002305 250.14131452]
 [  0.           0.           1.        ]]
畸变系数:
 [[ 0.01506212  0.76803888  0.00004892  0.00144843 -2.90071956]]
++++++++++相机标定完成++++++++++++++
+++++++++++手眼标定完成+++++++++++++++
旋转矩阵：
[[-0.99982957 -0.01821026  0.00303718]
 [ 0.01822104 -0.99982765  0.00355952]
 [ 0.00297184  0.00361425  0.99998905]]
平移向量：
[[0.03326139]
 [0.0855601 ]
 [0.02485716]]

 旋转矩阵：
[[-0.99981036 -0.01783498  0.00781984]
 [ 0.01783919 -0.99984076  0.00046898]
 [ 0.00781023  0.00060839  0.99996931]]
平移向量：
[[0.03139314]
 [0.08736614]
 [0.02635558]]

 旧深度

 
内参矩阵:
 [[904.99599938   0.         642.13768514]
 [  0.         905.53477292 372.51695536]
 [  0.           0.           1.        ]]
畸变系数:
 [[ 0.10377055 -0.19181945 -0.00006905  0.00166103 -0.06425022]]
有效图像编号: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
++++++++++相机标定完成++++++++++++++
+++++++++++手眼标定完成+++++++++++++++
旋转矩阵：
[[-0.99933658 -0.03624883  0.00352581]
 [ 0.03631033 -0.99915405  0.01930635]
 [ 0.00282299  0.01942157  0.9998074 ]]
平移向量：
[[0.03316422]
 [0.08138629]
 [0.01974935]]


 camera intrinsics: [ 1280x720  p[639.969 375.13]  f[910.507 910.957]  Inverse Brown Conrady [0 0 0 0 0] ]

'''
