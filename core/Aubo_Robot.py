# coding=utf8
import time
import copy
import numpy as np
import math
from core.jaw_control import JawController
from core.Camera import RSD435i
from core.robotcontrol import Auboi5Robot

def compute_step_stable_improve(delta,
                        max_step=0.08,   # 最大步长 1cm
                        min_step=0.001,     # 最小步长 0.5 mm
                        slow_radius=0.03,# 开始减速的距离 4cm
                        k_max=0.6,       # 最大增益
                        deadband=0.0001   # 0.5mm 死区
                        ):
    """
    稳定动态步长计算函数
    输入：
        delta: np.ndarray (3,) 目标-当前位置差
        max_step: 最大步长 (m)
        slow_radius: 距离阈值，越靠近增益越小
        k_max: 最大增益
        deadband: 死区，靠近目标小于此距离直接返回0
    输出：
        step: np.ndarray (3,) 实际移动步长
    """
    delta = np.asarray(delta, dtype=np.float64)
    d = np.linalg.norm(delta[:2])  # 只考虑 XY 平面距离

    # ---- 死区处理
    if d < deadband:
        return np.zeros_like(delta)

    # ---- 保持方向
    dir_vec = delta / d  # 方向向量

    # ---- 非线性增益（S 型函数）
    t = np.clip(d / slow_radius, 0.0, 1.0)
    gain = k_max * (3 * t**2 - 2 * t**3)  # smoothstep

    # ---- 步长计算
    step_len = max(gain * d, min_step)  # 保证最小步长
    step_len = min(step_len, max_step)
    # print("gain * d:", gain * d)
    step = dir_vec * step_len

    # ---- Z 轴可直接用 delta，不做平面增益缩放（根据需要）
    # step[2] = np.clip(delta[2], -max_step, max_step)

    return step


class Aubo_Robot(Auboi5Robot):
    def __init__(self, robot_host_ip="192.168.1.40", robot_port=8899,is_connect=True ,workspace_limits=None, #
                 is_use_jaw=False):
        super().__init__()
        # Init varibles
        if workspace_limits is None:
            workspace_limits = [[-0.35, 0], [0, 0.6], [0.38, 0.6]]
        self.workspace_limits = workspace_limits
        self.robot_host_ip = robot_host_ip
        self.robot_port = robot_port
        self.is_use_jaw = is_use_jaw
        self.is_connect=is_connect
        self.fixed_tcp_rpy = [-180.0, 0.0, 135]

        if (self.is_connect):
            self.create_context()
            self.connect(ip=self.robot_host_ip, port=self.robot_port)


        # jaw gripper configuration
        if (self.is_use_jaw):
            # jaw activate
            self.jaw = JawController()
            print("Activating jaw...")
            self.jaw.reset()
            time.sleep(1.5)

        # # Load camera pose (from running calibrate.py), intrinsics and depth scale
        self.cam_pose = np.loadtxt('./camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('./camera_depth_scale.txt', delimiter=' ')
        self.tool_pose = np.loadtxt('./tool_calibration_result.txt', delimiter=' ')

        # Default robot home joint configuration (the robot is up to air)
        self.home_joint_config = [-(90 / 360.0) * 2 * np.pi, -(-25 / 360.0) * 2 * np.pi,
                                  -(65 / 360.0) * 2 * np.pi, -(0 / 360.0) * 2 * np.pi,
                                  -(90 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi]
        
        self.start_joint_config = [-(46 / 360.0) * 2 * np.pi, -(32 / 360.0) * 2 * np.pi,
                                  -(104 / 360.0) * 2 * np.pi, -(-16 / 360.0) * 2 * np.pi,
                                  -(90 / 360.0) * 2 * np.pi, -(45 / 360.0) * 2 * np.pi]
        self.end_joint_config = [-(76 / 360.0) * 2 * np.pi, -(-18 / 360.0) * 2 * np.pi,
                                  -(56 / 360.0) * 2 * np.pi, -(-14 / 360.0) * 2 * np.pi,
                                  -(90 / 360.0) * 2 * np.pi, -(75 / 360.0) * 2 * np.pi]
        self.middle_joint_config = [-(63 / 360.0) * 2 * np.pi, -(20 / 360.0) * 2 * np.pi,
                                  -(98 / 360.0) * 2 * np.pi, -(-10 / 360.0) * 2 * np.pi,
                                  -(90 / 360.0) * 2 * np.pi, -(62 / 360.0) * 2 * np.pi]

        self.test_joint_config = [-(90 / 360.0) * 2 * np.pi, -(0 / 360.0) * 2 * np.pi,
                                  -(90 / 360.0) * 2 * np.pi, -(10 / 360.0) * 2 * np.pi,
                                  -(90 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi]
        
        self.cam1_joint_config = [-(55 / 360.0) * 2 * np.pi, -(2.7 / 360.0) * 2 * np.pi,
                                  -(82 / 360.0) * 2 * np.pi, -(-18 / 360.0) * 2 * np.pi,
                                  -(103 / 360.0) * 2 * np.pi, -(53 / 360.0) * 2 * np.pi]
        
        self.cam2_joint_config = [-(91 / 360.0) * 2 * np.pi, -(26 / 360.0) * 2 * np.pi,
                                  -(114 / 360.0) * 2 * np.pi, -(0 / 360.0) * 2 * np.pi,
                                  -(62 / 360.0) * 2 * np.pi, -(91 / 360.0) * 2 * np.pi]
        
        self.cam5_joint_config = [-(66 / 360.0) * 2 * np.pi, -(14 / 360.0) * 2 * np.pi,
                                  -(94 / 360.0) * 2 * np.pi, -(-8 / 360.0) * 2 * np.pi,
                                  -(90 / 360.0) * 2 * np.pi, -(65 / 360.0) * 2 * np.pi]
        
        self.cam3_joint_config = [-(74 / 360.0) * 2 * np.pi, -(2 / 360.0) * 2 * np.pi,
                                  -(86 / 360.0) * 2 * np.pi, -(-20 / 360.0) * 2 * np.pi,
                                  -(78 / 360.0) * 2 * np.pi, -(75 / 360.0) * 2 * np.pi]
        
        self.cam4_joint_config = [-(50 / 360.0) * 2 * np.pi, -(14 / 360.0) * 2 * np.pi,
                                  -(95 / 360.0) * 2 * np.pi, -(-0 / 360.0) * 2 * np.pi,
                                  -(111 / 360.0) * 2 * np.pi, -(57 / 360.0) * 2 * np.pi]

    def get_info(self):
        end_line_acc = self.get_end_max_line_acc()
        end_line_velc = self.get_end_max_line_velc()
        joint_acc = self.get_joint_maxacc()
        joint_velc = self.get_joint_maxvelc()
        print("end_max_line_acc:",end_line_acc)
        print("end_line_velc:",end_line_velc)
        print("joint_acc:",joint_acc)
        print("joint_velc:",joint_velc)

    def set_param(self):

        self.set_end_max_line_velc(0.1)
        self.set_end_max_line_acc(0.1)
        # 设置关节最大加速度
        self.set_joint_maxacc((1.5, 1.5, 1.5, 1.5, 1.5, 1.5))
        # 设置关节最大加速度
        self.set_joint_maxvelc((1.5, 1.5, 1.5, 1.5, 1.5, 1.5))

        self.get_info()

    def rpy2R(self,rpy): # [r,p,y] 单位rad
        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                          [0, math.sin(rpy[0]), math.cos(rpy[0])]])
        rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                          [0, 1, 0],
                          [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
        rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                          [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                          [0, 0, 1]])
        R = np.dot(rot_z, np.dot(rot_y, rot_x))
        return R

    def R2rpy(self,R):
    # assert (isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    def go_home(self):
        # self.move_joint(self.home_joint_config)
        self.move_joint(self.start_joint_config)

    def go_test(self):
        self.move_joint(self.test_joint_config)

    def move_around(self):
        self.move_line(self.start_joint_config)
        self.move_line(self.middle_joint_config)
    
    def go_collect(self): #制作数据集用的
        self.move_line(self.cam3_joint_config)
        self.move_line(self.cam2_joint_config)
        self.move_line(self.cam1_joint_config)
        self.move_line(self.cam4_joint_config)
        self.move_line(self.cam5_joint_config)

    def close_gripper(self):
        self.jaw.jaw_control(1)
        print("jaw had closed!")
        time.sleep(1.2)

    def open_gripper(self):
        self.jaw.jaw_control(0)
        print("gripper had opened!")
        time.sleep(1.2)

    def plane_grasp(self, position, yaw=0):
        rpy = [-180 ,0, 90 - yaw]
        # 判定抓取的位置是否处于工作空间
        for i in range(3):
            position[i] = min(max(position[i], self.workspace_limits[i][0]), self.workspace_limits[i][1])
        # 判定抓取的角度RPY是否在规定范围内 [-pi,pi]
        for i in range(3):
            if rpy[i] > 180:
                rpy[i] -= 2 * 180
            elif rpy[i] < -180:
                rpy[i] += 2 * 180
        print('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)' \
              % (position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]))

        # pre work
        grasp_home = [-0.1, 0.5, 0.4, -180, 0, 135]
        self.move_to_target_in_cartesian(grasp_home[:3],grasp_home[3:])

        if self.is_use_jaw:
            print("gripper open:")
            self.open_gripper()

        # Firstly, achieve pre-grasp position
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.1  # z axis
        # print(pre_position)
        self.move_to_target_in_cartesian(pre_position ,rpy)

        # Second，achieve grasp position
        self.move_to_target_in_cartesian(position ,rpy)
        if self.is_use_jaw:
            self.close_gripper()
        self.move_to_target_in_cartesian(pre_position ,rpy)
        # if (self.check_grasp()):
        #     print("Check grasp fail! ")
        #     self.move_to_target_in_cartesian(grasp_home)
        #     return False
        # Third,put the object into box
        box_position = [0.3, 0.3, 0.35, -180, 0, 135]
        self.move_to_target_in_cartesian(box_position[:3],box_position[3:])
        box_position[2] = 0.25  # down to the 10cm
        self.move_to_target_in_cartesian(box_position[:3],box_position[3:])
        if self.is_use_jaw:
            self.open_gripper()
        box_position[2] = 0.35
        self.move_to_target_in_cartesian(box_position[:3],box_position[3:])
        self.move_to_target_in_cartesian(grasp_home[:3],grasp_home[3:])
        print("grasp success!")
        return True

    def plane_push(self, position, move_orientation=0, length=0.1):
        for i in range(2):
            position[i] = min(max(position[i], self.workspace_limits[i][0] + 0.1), self.workspace_limits[i][1] - 0.1)
        position[2] = min(max(position[2], self.workspace_limits[2][0]), self.workspace_limits[2][1])
        print('Executing: push at (%f, %f, %f) and the orientation is %f' % (position[0], position[1], position[2],
                                                                             move_orientation))

        push_home = [0.4, 0, 0.4, -np.pi, 0, 0]
        self.move_to_target_in_cartesian(push_home[:3],push_home[3:])  # pre push position(push home)
        # self.close_gripper()

        self.move_to_target_in_cartesian([position[0], position[1], position[2] + 0.1],[ -np.pi, 0, 0])
        self.move_to_target_in_cartesian([position[0], position[1], position[2]],[ -np.pi, 0, 0])

        # compute the destination pos
        destination_pos = [position[0] + length * math.cos(move_orientation),
                           position[1] + length * math.sin(move_orientation), position[2]]
        self.move_to_target_in_cartesian(destination_pos , [-np.pi, 0, 0])
        self.move_to_target_in_cartesian([destination_pos[0], destination_pos[1], destination_pos[2] + 0.1],[ -np.pi, 0, 0])

        # go back push-home
        self.move_to_target_in_cartesian(push_home[:3],push_home[3:])

    def grasp(self, position, rpy=None, k_acc=0.8, k_vel=0.8, speed=255, force=125):

        # 判定抓取的位置是否处于工作空间
        if rpy is None:
            rpy = [-np.pi, 0, 0]
        for i in range(3):
            position[i] = min(max(position[i], self.workspace_limits[i][0]), self.workspace_limits[i][1])
        # 判定抓取的角度RPY是否在规定范围内 [0.5*pi,1.5*pi]
        for i in range(3):
            if rpy[i] > np.pi:
                rpy[i] -= 2 * np.pi
            elif rpy[i] < -np.pi:
                rpy[i] += 2 * np.pi
        print('Executing: grasp at (%f, %f, %f) by the RPY angle (%f, %f, %f)' \
              % (position[0], position[1], position[2], rpy[0], rpy[1], rpy[2]))

        # pre work
        grasp_home = [0.4, 0, 0.4, -np.pi, 0, 0]  # you can change me
        self.move_to_target_in_cartesian(grasp_home[:3],grasp_home[3:])
        self.open_gripper()

        # Firstly, achieve pre-grasp position
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.1  # z axis
        print(pre_position)
        self.move_to_target_in_cartesian(pre_position , rpy)

        # Second，achieve grasp position
        self.move_to_target_in_cartesian(position ,rpy)
        self.close_gripper()
        self.move_to_target_in_cartesian(pre_position , rpy)
        # if (self.check_grasp()):
        #     print("Check grasp fail! ")
        #     self.move_j_p(grasp_home)
        #     return False
        # Third,put the object into box
        box_position = [0.63, 0, 0.25, -np.pi, 0, 0]  # you can change me!
        self.move_to_target_in_cartesian(box_position[:3],box_position[3:])
        box_position[2] = 0.1  # down to the 10cm
        self.move_to_target_in_cartesian(box_position[:3],box_position[3:])
        self.open_gripper()
        box_position[2] = 0.25
        self.move_to_target_in_cartesian(box_position[:3],box_position[3:])
        self.move_to_target_in_cartesian(grasp_home[:3],grasp_home[3:])
        print("grasp success!")

    def align_to_target(self, position, yaw=-90, z_offset=0.0):
        """
        仅用于对准目标点（视觉点击）
        - 不抓取
        - 不下探
        """


        # ===== 2. 平面朝下对准姿态 =====
        # Z轴向下，绕Z旋转 yaw
        rpy = [-180, 0, yaw]

        # RPY 角度归一化到 [-180, 180]
        for i in range(3):
            if rpy[i] > 180:
                rpy[i] -= 360
            elif rpy[i] < -180:
                rpy[i] += 360

        # ===== 3. 可选高度偏移（对准而不贴近） =====
        target_position = position.copy()
        target_position[2] += z_offset

        # ===== 1. 工作空间合法性判定（拒绝非法） =====
        for i in range(3):
            low, high = self.workspace_limits[i]
            if not (low <= target_position[i] <= high):
                print(f"[Reject] Target out of workspace on axis {i}: {target_position[i]:.3f}")
                return False

        print(
            "Aligning to target: "
            f"pos=({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}), "
            f"rpy=({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f})"
        )

        # ===== 4. 执行运动 =====
        self.move_to_target_in_cartesian(target_position, rpy)
        return True

    def align_to_target_line_1(self, position, z_offset=0.0):
        """
        视觉对准目标
        - 直线运动（move_line）
        - 姿态保持不变
        - 显式 IK，失败可控
        """

        # ===== 1. 构造目标位置（只改位置，不动姿态）=====
        target_pos = position.copy()
        print("target_pos before z offset:", target_pos)
        target_pos[2] += z_offset

        # ===== 2. 工作空间检查 =====
        for i in range(3):
            low, high = self.workspace_limits[i]
            if not (low <= target_pos[i] <= high):
                print(f"[Reject] Target: {target_pos[i]:.3f} out of workspace on axis {i}:[{low:.3f}, {high:.3f}]")
                return False

        # ===== 3. 获取当前 TCP 状态 =====
        current_wp = self.get_current_waypoint()
        current_joint = current_wp['joint']
        current_ori   = current_wp['ori']   #保持当前姿态
        cur_pos = current_wp['pos']

        delta = target_pos - cur_pos

        # 到位判断
        if np.linalg.norm(delta[:2]) <= 0.001:
            print(
            "Aligned: "
            f"pos=({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
        )
            return True

        # step = compute_step_stable_improve(delta)
        # print("step:",step)
        # next_pos = cur_pos + step
        next_pos = target_pos

        print(
            "Aligning (line by step): "
            f"pos=({next_pos[0]:.3f}, {next_pos[1]:.3f}, {next_pos[2]:.3f})"
        )

        # ===== 4. 显式 IK =====
        ik_result = self.inverse_kin(
            current_joint,
            next_pos,
            current_ori
        )

        if ik_result is None:
            print("[WARN] IK failed, skip align_to_target")
            return False

        # ===== 5. 直线运动 =====
        self.move_line(ik_result['joint'])

        return True

    def align_to_target_line(
        self,
        position,
        z_offset=0.3,
        use_fixed_z=False
    ):
        """
        视觉对准目标（直线运动）
        - move_line
        - 姿态保持不变
        - 显式 IK，失败可控

        Args:
            position: 视觉给出的目标位置 (x, y, z)
            z_offset: 
                - use_fixed_z=False 时：在 position.z 基础上的偏移
                - use_fixed_z=True  时：直接作为目标 z
            use_fixed_z: 是否固定 Z 轴高度
        """

        # ===== 1. 构造目标位置（只改位置，不动姿态）=====
        target_pos = position.copy()
        # print("target_pos before z process:", target_pos)

        if use_fixed_z:
            # ✅ 固定 Z 轴（典型：插孔 / 拧紧 / 贴合）
            target_pos[2] = z_offset
            # print(f"[Z FIXED] z = {z_offset:.3f}")
        else:
            # ✅ 视觉 Z + offset（例如深度相机）
            target_pos[2] += z_offset
            # print(f"[Z OFFSET] z += {z_offset:.3f}")

        # ===== 2. 工作空间检查 =====
        for i in range(3):
            low, high = self.workspace_limits[i]
            if not (low <= target_pos[i] <= high):
                print(
                    f"[Reject] Target: {target_pos[i]:.3f} "
                    f"out of workspace on axis {i}: [{low:.3f}, {high:.3f}]"
                )
                return False

        # ===== 3. 获取当前 TCP 状态 =====
        current_wp = self.get_current_waypoint()
        current_joint = current_wp['joint']
        current_ori   = current_wp['ori']   # 保持当前姿态
        cur_pos       = current_wp['pos']

        delta = target_pos - cur_pos

        # ===== 4. 到位判断（只关心 XY）=====
        if np.linalg.norm(delta[:2]) <= 0.001:
            print(
                "Aligned: "
                f"pos=({target_pos[0]:.3f}, "
                f"{target_pos[1]:.3f}, "
                f"{target_pos[2]:.3f})"
            )
            return True

        # step = compute_step_stable_improve(delta)
        # print("step:",step)
        # next_pos = cur_pos + step

        # 直线目标（你目前是一步到位）
        next_pos = target_pos

        print(
            "Aligning (line): "
            f"pos=({next_pos[0]:.3f}, "
            f"{next_pos[1]:.3f}, "
            f"{next_pos[2]:.3f})"
        )

        # ===== 5. 显式 IK =====
        ik_result = self.inverse_kin(
            current_joint,
            next_pos,
            current_ori
        )

        if ik_result is None:
            print("[WARN] IK failed, skip align_to_target")
            return False

        # ===== 6. 直线运动 =====
        self.move_line(ik_result['joint'])

        return True

    def compute_target2base(self, cam2target, use_tool=False):
        """
        根据相机点击点计算机器人目标末端位置，并对齐末端执行器
        
        参数:
            cam2target: np.array or list, [x, y, z] 相机坐标系中的目标点
            use_tool: bool, 是否考虑工具坐标系
            z_offset: float, 对齐时末端在Z方向的偏移量
        
        返回:
            target_pos: np.array, 相对于机器人基座的末端目标位置
        """

        cam2target = np.asarray(cam2target).reshape(3,1)#先保证是列向量
        # 获取相机在法兰上的位姿
        flange2camera = self.cam_pose  

        # 当前法兰在基座上的位姿
        current_point = self.get_current_waypoint()
        base2flange = np.eye(4)
        base2flange[:3, 3] = current_point['pos']
        rpy = self.quaternion_to_rpy(current_point['ori'])
        base2flange[:3, :3] = self.rpy2R(rpy)
        # print("base2flange:", base2flange[:3, 3].flatten())

        # 基座到相机
        base2camera = base2flange @ flange2camera
        # print("base2camera:", base2camera[:3, 3].flatten())

        # 基座到物体（先旋转再平移）
        base2obj = base2camera[:3, :3] @ cam2target + base2camera[:3, 3:4] 
        base2obj_position = base2obj.flatten()
        # print("base2obj_position:", base2obj_position)

        # 法兰到工具（如果使用工具）
        flange2tool = self.tool_pose if use_tool else flange2camera

        # 生成目标末端位姿
        base2target = np.eye(4)
        base2target[:3, :3] = base2camera[:3, :3]  # 保持与相机相同的姿态
        base2target[:3, 3] = base2obj_position

        # 计算期望法兰位姿
        base2flange_expect = base2target @ np.linalg.inv(flange2tool)
        target_pos = base2flange_expect[:3, 3]

        return target_pos


    def align_to_target_line_stepwise(
    self,
    position,
    z_offset=0.3,
    use_fixed_z=False,
    max_step_xy=0.005,
    xy_deadband=0.001
    ):
        """
        视觉对准目标（分步直线运动）
        - 每次只走向目标的一小步
        - 姿态保持不变
        - 显式 IK，失败可控

        Args:
            position: base系目标位置 (x, y, z)
            z_offset:
                - use_fixed_z=False 时：在 position.z 基础上的偏移
                - use_fixed_z=True  时：直接作为目标 z
            use_fixed_z: 是否固定 Z
            max_step_xy: 单次命令在 XY 平面的最大步长（m）
            xy_deadband: XY 死区（m）
        """
        import numpy as np

        target_pos = np.array(position, dtype=float).copy()

        if use_fixed_z:
            target_pos[2] = z_offset
        else:
            target_pos[2] += z_offset

        # 工作空间检查
        for i in range(3):
            low, high = self.workspace_limits[i]
            if not (low <= target_pos[i] <= high):
                print(
                    f"[Reject] Target {target_pos[i]:.3f} out of workspace on axis {i}: "
                    f"[{low:.3f}, {high:.3f}]"
                )
                return False

        current_wp = self.get_current_waypoint()
        current_joint = current_wp["joint"]
        current_ori = current_wp["ori"]
        cur_pos = np.array(current_wp["pos"], dtype=float)

        delta = target_pos - cur_pos
        delta_xy = delta[:2]
        dist_xy = np.linalg.norm(delta_xy)

        if dist_xy <= xy_deadband:
            print(
                f"[Aligned-step] target=({target_pos[0]:.3f}, "
                f"{target_pos[1]:.3f}, {target_pos[2]:.3f})"
            )
            return True

        if dist_xy > max_step_xy:
            step_xy = delta_xy / dist_xy * max_step_xy
        else:
            step_xy = delta_xy

        next_pos = cur_pos.copy()
        next_pos[0] += step_xy[0]
        next_pos[1] += step_xy[1]
        next_pos[2] = target_pos[2]

        # 再次检查工作空间
        for i in range(3):
            low, high = self.workspace_limits[i]
            if not (low <= next_pos[i] <= high):
                print(
                    f"[Reject-next] next_pos[{i}]={next_pos[i]:.3f} "
                    f"out of workspace: [{low:.3f}, {high:.3f}]"
                )
                return False

        print(
            "[Step Align] "
            f"cur=({cur_pos[0]:.3f},{cur_pos[1]:.3f},{cur_pos[2]:.3f}) -> "
            f"next=({next_pos[0]:.3f},{next_pos[1]:.3f},{next_pos[2]:.3f}) "
            f"target=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) "
            f"dist_xy={dist_xy:.4f}, step_xy={np.linalg.norm(step_xy):.4f}"
        )

        ik_result = self.inverse_kin(current_joint, next_pos, current_ori)
        if ik_result is None:
            print("[WARN] IK failed, skip align_to_target_line_stepwise")
            return False

        self.move_line(ik_result["joint"])
        return True


if __name__ == "__main__":
    # 系统初始化
    Aubo_Robot.initialize()

    robot = Aubo_Robot()

    robot.go_home()

    # robot.plane_grasp(position=[0,0.4,0.3])

