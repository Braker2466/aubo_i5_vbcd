import serial   # pip install pyserial
from time import sleep
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JawController:
    def __init__(self):
        self.ser = serial.Serial()
        self.is_connected = False

        # 预定义的命令字节序列
        self.COMMANDS = {
            'close': bytearray([0x7B, 0x01, 0x02, 0x01, 0x20, 0x49, 0x20, 0x00, 0xC8, 0xF8, 0x7D]),
            'open': bytearray([0x7B, 0x01, 0x02, 0x00, 0x20, 0x49, 0x20, 0x00, 0xC8, 0xF9, 0x7D])
        }

    def port_open(self, port='/dev/ttyACM0', baudrate=115200, timeout=1):
        """打开串口连接"""
        try:
            self.ser.port = port
            self.ser.baudrate = baudrate
            self.ser.bytesize = 8
            self.ser.stopbits = 1
            self.ser.parity = 'N'
            self.ser.timeout = timeout

            if not self.ser.is_open:
                self.ser.open()

            self.is_connected = True
            logger.info(f"串口打开成功: {port}")
            return True

        except Exception as e:
            logger.error(f"串口打开失败: {e}")
            self.is_connected = False
            return False

    def port_close(self):
        """关闭串口连接"""
        try:
            if self.ser.is_open:
                self.ser.close()
            self.is_connected = False
            logger.info("串口关闭成功")
            return True
        except Exception as e:
            logger.error(f"串口关闭失败: {e}")
            return False

    def send_command(self, command_data):
        """发送命令数据"""
        if not self.is_connected or not self.ser.is_open:
            logger.error("串口未连接，发送失败")
            return False

        try:
            self.ser.write(command_data)
            logger.info(f"命令发送成功: {command_data.hex()}")
            return True
        except Exception as e:
            logger.error(f"命令发送失败: {e}")
            return False

    def jaw_control(self, enable, delay=0.5):
        """
        控制夹爪

        Args:
            enable: 1-闭合, 0-张开
            delay: 执行后的延迟时间（秒）
        """
        # 确保串口已连接
        if not self.is_connected:
            if not self.port_open():
                return False

        try:
            if enable == 1:
                command = self.COMMANDS['close']
                action = "闭合"
            elif enable == 0:
                command = self.COMMANDS['open']
                action = "张开"
            else:
                logger.error(f"无效的控制参数: {enable}")
                return False

            if self.send_command(command):
                logger.info(f"夹爪{action}命令执行成功")
                sleep(delay)  # 等待动作完成
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"夹爪控制异常: {e}")
            return False

    def reset(self):

        # 控制夹爪闭合
        self.jaw_control(1)
        sleep(1)  # 等待1秒
        # 控制夹爪张开
        self.jaw_control(0)
        self.port_close()


    def __del__(self):
        """析构函数，确保串口关闭"""
        self.port_close()

#此文件是师兄留下的夹爪使用，我保留了，用的时候自己看着改。
# 使用示例
if __name__ == '__main__':
    jaw = JawController()
    jaw.reset()

    try:
        # 控制夹爪闭合
        if jaw.jaw_control(1):
            print("夹爪闭合成功")

        sleep(1)  # 等待1秒

        # 控制夹爪张开
        if jaw.jaw_control(0):
            print("夹爪张开成功")

    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        jaw.port_close()

