import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


class ToolCalibration:
    def __init__(self, csv_file="tool_data.csv"):
        """
        工具坐标系标定类
        标定原理：通过机器人在不同姿态下触探同一固定点，计算工具坐标系相对于法兰的变换

        Args:
            csv_file: 包含机器人位姿数据的CSV文件
                      每行格式: [x, y, z, rx, ry, rz]
                      x,y,z: 位置(m), rx,ry,rz: 欧拉角(度)
        """
        self.csv_file = csv_file
        self.poses = None
        self.tool_T = None
        self.load_data()

    def load_data(self):
        try:
            with open(self.csv_file) as file:
                self.poses = pd.read_csv(file, header=None)
                self.poses = np.array(self.poses)

            print(f"成功加载 {len(self.poses)} 个标定点数据")

        except FileNotFoundError:
            print(f"错误: 文件 {self.csv_file} 不存在")
            self.poses = np.array([])
        except Exception as e:
            print(f"加载数据时出错: {e}")
            self.poses = np.array([])

    def pose_to_matrix(self, pose):
        """
        将位姿向量转换为齐次变换矩阵
        Args:
            pose: [x, y, z, rx, ry, rz]
        Returns:
            4x4齐次变换矩阵
        """
        if len(pose) < 6:
            raise ValueError(f"位姿数据需要6个元素，得到{len(pose)}个")

        # 创建旋转矩阵（注意：scipy的from_euler使用'xyz'外旋顺序）
        r = R.from_euler('xyz', pose[3:6], degrees=True)
        rotation_matrix = r.as_matrix()  # 新版本使用as_matrix()

        # 创建齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = pose[:3]

        return T

    def calibrate_translation(self, pose_indices=None):
        """
        TCP标定工具坐标系原点（平移部分）
        原理：在不同姿态下，工具尖端指向同一固定点

        Args:
            pose_indices: 用于平移标定的数据索引列表，使用前4个

        Returns:
            工具坐标系原点在法兰坐标系下的坐标 (3x1)
        """
        if pose_indices is None:
            pose_indices = range(4)  # 默认使用前4个点

        if len(pose_indices) < 3:
            raise ValueError("至少需要3个点进行平移标定")

        # 准备数据
        A_list = []
        b_list = []

        for i in range(len(pose_indices) - 1):
            idx1 = pose_indices[i]
            idx2 = pose_indices[i + 1]

            # 获取两个相邻的位姿
            pose1 = self.poses[idx1]
            pose2 = self.poses[idx2]

            # 转换为变换矩阵
            T1 = self.pose_to_matrix(pose1)  # 法兰到基座
            T2 = self.pose_to_matrix(pose2)

            # 提取旋转和平移
            R1 = T1[:3, :3]
            R2 = T2[:3, :3]
            t1 = T1[:3, 3].reshape(3, 1)
            t2 = T2[:3, 3].reshape(3, 1)

            # 构建方程: (R1 - R2) * p_tool = t2 - t1
            # 其中 p_tool 是工具尖端在法兰坐标系下的坐标
            A_list.append(R1 - R2)
            b_list.append(t2 - t1)

        # 使用最小二乘法求解
        A = np.vstack(A_list)
        b = np.vstack(b_list)

        # 求解 p_tool
        p_tool, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        print(f"平移标定完成，残差: {residuals[0] if len(residuals) > 0 else 'N/A'}")
        print(f"工具尖端位置 (法兰坐标系): [{p_tool[0, 0]:.3f}, {p_tool[1, 0]:.3f}, {p_tool[2, 0]:.3f}] m")

        return p_tool

    def calibrate_rotation(self, p_tool, pose_indices=None):
        """
        TCF标定工具坐标系姿态（旋转部分）
        原理：通过工具在不同方向的指向确定坐标系

        Args:
            p_tool: 工具尖端位置 (3x1)
            pose_indices: 用于旋转标定的数据索引列表，默认使用后3个

        Returns:
            工具坐标系到法兰坐标系的旋转矩阵 (3x3)
        """
        if pose_indices is None:
            pose_indices = range(3, 6)  # 默认使用第4,5,6个点

        if len(pose_indices) < 3:
            raise ValueError("至少需要3个点进行旋转标定")

        # 计算工具尖端在基座坐标系中的位置
        tool_points_base = []

        for idx in pose_indices:
            pose = self.poses[idx]
            T = self.pose_to_matrix(pose)
            R_f = T[:3, :3]  # 法兰到基座的旋转
            t_f = T[:3, 3].reshape(3, 1)  # 法兰到基座的平移

            # 工具尖端在基座坐标系中的位置
            p_base = R_f @ p_tool + t_f
            tool_points_base.append(p_base.flatten())

        tool_points_base = np.array(tool_points_base)

        # 第4点作为原点
        origin = tool_points_base[0]

        # 计算X轴方向（指向第二个点）
        x_vec = tool_points_base[1] - origin
        x_axis = x_vec / np.linalg.norm(x_vec)

        # 计算Z轴方向（指向第三个点）
        z_vec = tool_points_base[2] - origin
        z_axis = z_vec / np.linalg.norm(z_vec)

        # 计算Y轴方向（垂直于XZ平面）
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # 重新计算Z轴确保正交
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # 构建工具坐标系在基座中的旋转矩阵
        tool_rot_in_base = np.column_stack([x_axis, y_axis, z_axis])

        # 计算工具坐标系到法兰坐标系的旋转
        # 第一个姿态的法兰到基座旋转
        first_pose = self.pose_to_matrix(self.poses[pose_indices[0]])
        R_f_base = first_pose[:3, :3]

        # 工具到法兰的旋转
        R_tool_f = R_f_base.T @ tool_rot_in_base

        # 确保旋转矩阵是正交的
        U, S, Vt = np.linalg.svd(R_tool_f)
        R_tool_f = U @ Vt

        # 检查行列式，确保是右手系
        if np.linalg.det(R_tool_f) < 0:
            R_tool_f = U @ np.diag([1, 1, -1]) @ Vt

        return R_tool_f

    def calibrate(self):
        """执行完整的工具坐标系标定"""
        if len(self.poses) < 6:
            raise ValueError("至少需要6个标定点数据")

        print("开始工具坐标系标定...")
        print("=" * 50)

        # 1. 标定平移
        print("\n1. 平移标定")
        print("-" * 30)
        p_tool = self.calibrate_translation()

        # 2. 标定旋转
        print("\n2. 旋转标定")
        print("-" * 30)
        R_tool = self.calibrate_rotation(p_tool)

        # 3. 组合成齐次变换矩阵
        print("\n3. 生成变换矩阵")
        print("-" * 30)
        self.tool_T = np.eye(4)
        self.tool_T[:3, :3] = R_tool
        self.tool_T[:3, 3] = p_tool.flatten()

        # 4. 转换为欧拉角（方便理解）
        r = R.from_matrix(R_tool)
        euler_angles = r.as_euler('xyz', degrees=True)

        print(f"工具坐标系变换矩阵:")
        print(self.tool_T)
        print(f"\n工具位置 (法兰坐标系): [{p_tool[0, 0]:.3f}, {p_tool[1, 0]:.3f}, {p_tool[2, 0]:.3f}] m")
        print(f"工具欧拉角 (xyz, 度): [{euler_angles[0]:.3f}, {euler_angles[1]:.3f}, {euler_angles[2]:.3f}]")

        return self.tool_T

    def validate_calibration(self, test_indices=None):
        """验证标定结果"""
        if self.tool_T is None:
            print("请先执行标定")
            return

        if test_indices is None:
            test_indices = range(len(self.poses[6:]))  # 7-8验证点

        print("\n标定结果验证")
        print("=" * 50)

        errors = []
        for idx in test_indices:
            if idx >= len(self.poses):
                break

            pose = self.poses[idx]
            T_base_flange = self.pose_to_matrix(pose)

            # 计算工具尖端在基座中的理论位置
            T_base_tool = T_base_flange @ self.tool_T
            tool_tip_base = T_base_tool[:3, 3]

            # 理论上所有点应该接近同一个位置
            errors.append(tool_tip_base)

        errors = np.array(errors)
        center = np.mean(errors, axis=0)
        deviations = errors - center

        max_dev = np.max(np.linalg.norm(deviations, axis=1))
        avg_dev = np.mean(np.linalg.norm(deviations, axis=1))

        print(f"工具尖端理论位置中心: [{center[0]:.5f}, {center[1]:.5f}, {center[2]:.5f}] m")
        print(f"最大偏差: {max_dev:.5f} m")
        print(f"平均偏差: {avg_dev:.5f} m")

        if avg_dev < 0.0010:  # 可根据精度要求调整
            print("✓ 标定结果良好")
        else:
            print("⚠ 标定偏差较大，请检查数据质量")


# 使用示例
if __name__ == "__main__":
    # 创建标定器
    calibrator = ToolCalibration("tool_data.csv")

    # 执行标定
    if len(calibrator.poses) >= 6:
        tool_T = calibrator.calibrate()

        # 验证标定结果
        calibrator.validate_calibration()

        # 保存标定结果
        np.savetxt("tool_calibration_result.txt", tool_T, fmt='%.6f')
        print("\n标定结果已保存到 tool_calibration_result.txt")
    else:
        print("数据不足，无法进行标定")