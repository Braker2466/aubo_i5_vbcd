from core.robotcontrol import *
import numpy as np
import math
"""

机械臂轨距运动, add_waypoint()添加路点后move_track()沿路点进行运动

"""

def rpy2R(rpy): # [r,p,y] 单位rad
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

def R2rpy(R):
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


def test_process_demo():
    # 初始化logger
    logger_init()

    # 启动测试
    logger.info("{0} test beginning...".format(Auboi5Robot.get_local_time()))

    # 系统初始化
    Auboi5Robot.initialize()

    # 创建机械臂控制类
    robot = Auboi5Robot()

    # 创建上下文
    handle = robot.create_context()

    # 打印上下文
    logger.info("robot.rshd={0}".format(handle))

    try:

        # time.sleep(0.2)
        # process_get_robot_current_status = GetRobotWaypointProcess()
        # process_get_robot_current_status.daemon = True
        # process_get_robot_current_status.start()
        # time.sleep(0.2)

        queue = Queue()

        p = Process(target=runWaypoint, args=(queue,))
        p.start()
        time.sleep(5)
        print("process started.")

        # 链接服务器
        #ip = 'localhost'
        # ip = '192.168.174.128'  # 虚拟机ip
        ip = '192.168.65.100'  # 真机ip
        port = 8899
        result = robot.connect(ip, port)

        if result != RobotErrorType.RobotError_SUCC:
            logger.info("connect server{0}:{1} failed.".format(ip, port))
        else:
            robot.enable_robot_event()
            robot.init_profile()
            joint_maxvelc = (2.596177/15, 2.596177/15, 2.596177/15, 3.110177/15, 3.110177/15, 3.110177/15)
            joint_maxacc = (17.308779/5, 17.308779/5, 17.308779/5, 17.308779/5, 17.308779/5, 17.308779/5)
            robot.set_joint_maxacc(joint_maxacc)
            robot.set_joint_maxvelc(joint_maxvelc)
            robot.set_arrival_ahead_blend(0.05)


            time.sleep(1)

            home_radian = (-90/180.0*pi, 0, -90/180.0*pi, 0, -90/180.0*pi, 0)
            robot.move_joint(home_radian, True)

            waypoint1 = robot.get_current_waypoint()
            print(f"tcp pos:{waypoint1['pos']}")   # [-0.12149997497522927, 0.47850000120206204, 0.4124999790841]
            print(f"tcp ori:{waypoint1['ori']}")  # [2.2351741790771484e-08, 2.2351741790771484e-08, 0.9999999999999992, -2.2351741790771484e-08]


            tool_param={"pos":(0.00290,-0.00495,0.17894), "ori":(0.9263080761737628,-0.0036711989667299934,0.0017363499053789848,0.3767450801303387)}
            tool_pos_on_base = robot.base_to_base_additional_tool(waypoint1['pos'],
                                                                  waypoint1['ori'],
                                                                  tool_param)
            print(f"tcp pos with tool:{tool_pos_on_base['pos']}")  # [-0.12439996719727034, 0.47354999333246056, 0.23355997917574234]
            print(f"tcp ori with tool:{tool_pos_on_base['ori']}")  # [-0.0017363206978136029, 0.3767451007916901, 0.9263080678737214, 0.00367118672185025]

            ####################       ######################
            base2end = np.eye(4)
            base2end[:3, 3] = waypoint1['pos']
            rpy = robot.quaternion_to_rpy(waypoint1['ori'])
            base2end[:3, :3] = rpy2R(rpy)

            end2tool = np.loadtxt('./tool_calibration_result.txt', delimiter=' ')

            base2tool = base2end @ end2tool
            print(f"base2end with tool:{base2tool[:3,3]}")  # [-0.12439997  0.47355195  0.23355898]
            t= R2rpy(base2tool[:3,:3])
            p= robot.rpy_to_quaternion(t)
            print(f"base2end ori with tool:{p}")  # [-0.0017362576395880626, 0.3767452326858757, 0.9263080145517288, 0.0036711353977405157]


            # base2end = base2tool @ np.linalg.inv(end2tool)

            # 断开服务器链接
            robot.disconnect()

    except KeyboardInterrupt:
        robot.move_stop()

    except RobotError as e:
        logger.error("robot Event:{0}".format(e))



    finally:
        # 断开服务器链接
        if robot.connected:
            # 断开机械臂链接
            robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()
        print("run end-------------------------")

if __name__ == '__main__':
    test_process_demo()
    logger.info("test completed")