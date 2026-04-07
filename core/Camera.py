import numpy as np
import pyrealsense2 as rs
import time
import cv2

class RSD435i(object):

    def __init__(self,width=640,height=480,fps=15,default_depth=None):
        self.im_height = height
        self.im_width = width
        self.fps = fps
        self.default_depth = default_depth
        self.intrinsics = None
        self.scale = None
        self.pipeline = None
        self.connect()
        # self.cam_intrinsics = np.array([607.005, 0, 319.980, 0, 607.304, 250.081, 0, 0, 1]).reshape(3, 3)
        # self.cam_depth_scale = np.loadtxt('./camera_depth_scale.txt', delimiter=' ')
        # color_img, depth_img = self.get_data()
        #print(color_img, depth_img)

    def connect(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        # Start streaming
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = self.get_intrinsics(rgb_profile)
        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        print("camera depth scale:",self.scale)
        print("D435i have connected ...")

    def pixel_to_camera(self, x, y, depth_image, vx_px=0.0, vy_px=0.0, fps=30):
        """
        像素坐标 -> 相机坐标系 (单位: 米)，并转换像素速度为 m/s

        Args:
            x, y: 像素坐标
            depth_image: 对齐彩色图的深度图 (H, W, 1)
            vx_px, vy_px: 像素速度 (px/frame)
            fps: 帧率

        Returns:
            np.ndarray (5,): [X, Y, Z, VX, VY] in camera frame
            若坐标越界或深度无效，返回 None
        """

        # -------- 1. 转 int（Kalman 输出常为 float）
        x_int = int(round(x))
        y_int = int(round(y))

        # -------- 2. 图像边界检查
        h, w = depth_image.shape[:2]
        if x_int < 0 or x_int >= w or y_int < 0 or y_int >= h:
            return None

        # -------- 3. 读取深度（mm -> m）
        z = depth_image[y_int, x_int, 0] * self.scale
        if not np.isfinite(z) or z <= 0.06:
            if self.default_depth is not None:
                z = self.default_depth
            else:
                return None

        # -------- 4. 相机内参
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]

        # -------- 5. 反投影
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy

        # -------- 6. 像素速度 -> m/s
        VX = vx_px * z / fx * fps
        VY = vy_px * z / fy * fps

        return np.array([X, Y, z, VX, VY], dtype=np.float32)


    def get_data(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()

        # align
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # no align
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.float32)
        # depth_image *= self.scale
        depth_image = np.expand_dims(depth_image, axis=2)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def plot_image(self):
        color_image,depth_image = self.get_data()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # cv2.imwrite('color_image.png', color_image)
        cv2.waitKey(5000)

    def get_intrinsics(self,rgb_profile):
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        print("camera intrinsics:", raw_intrinsics)
        # camera intrinsics form is as follows.
        #[[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        # intrinsics = np.array([615.284,0,309.623,0,614.557,247.967,0,0,1]).reshape(3,3) #640 480
        intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)

        return intrinsics

class USBCamera(object):
    """
    USB 普通相机版本：
    - 仅提供彩色图
    - 深度图用常值 default_depth (米) 填充，shape 与 RealSense 对齐 (H, W, 1)
    - 保持 get_data() 的输入输出格式一致
    """

    def __init__(self, width=640, height=480, fps=30, default_depth=0.50, device_index=0):
        self.im_height = height
        self.im_width = width
        self.fps = fps
        self.default_depth = default_depth
        self.device_index = device_index

        self.intrinsics = None  # 3x3
        self.scale = 1.0        # USB 无 depth scale，保留字段占位
        self.cap = None

        self.connect()

    def connect(self):
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"USB camera open failed. device_index={self.device_index}")

        # 尝试设置分辨率与帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.im_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.im_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 读取一次，确认真实分辨率
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("USB camera read failed right after open().")

        h, w = frame.shape[:2]
        self.im_width = w
        self.im_height = h

        # USB 相机没有内参接口：给一个“可用的近似值”，用于保持字段存在
        # 如果你做视觉伺服/几何测量，建议你后面用标定结果替换这里的 intrinsics
        fx = 0.9 * w
        fy = 0.9 * w
        cx = w / 2.0
        cy = h / 2.0
        self.intrinsics = np.array([[fx, 0,  cx],
                                    [0,  fy, cy],
                                    [0,  0,  1]], dtype=np.float32)

        print("USB camera connected ...")
        print(f"resolution: {w}x{h}, fps(set): {self.fps}")
        print("camera intrinsics(approx):", self.intrinsics)

    def get_data(self):
        """
        Returns:
            color_image: np.ndarray uint8, (H, W, 3) BGR
            depth_image: np.ndarray float32, (H, W, 1) 常值深度(米)
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("USB camera not opened.")

        ok, color_image = self.cap.read()
        if not ok or color_image is None:
            return None, None

        h, w = color_image.shape[:2]

        # 深度常值（米），对齐 RealSense 输出 shape: (H, W, 1), dtype=float32
        z = float(self.default_depth) if self.default_depth is not None else 0.5
        depth_image = np.full((h, w, 1), z, dtype=np.float32)

        return color_image, depth_image

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

class FakeCamera(object):
    """
    视频文件模拟相机：
    - 输入：mp4 文件
    - 输出接口与 RealSense Camera 类一致
    - 支持控制“算法看到的帧率”和播放速度倍率
    """

    def __init__(self,
                 video_path,
                 width=None,
                 height=None,
                 fps=30,                 # 输出给算法的帧率
                 playback_speed=1.0,     # 播放倍率 (2.0=2倍速, 0.5=慢放)
                 default_depth=0.5):     # 常值深度 (米)

        self.video_path = video_path
        self.target_fps = fps
        self.playback_speed = playback_speed
        self.default_depth = default_depth

        self.cap = None
        self.intrinsics = None
        self.scale = 1.0

        self.im_width = width
        self.im_height = height

        self._frame_interval = 1.0 / self.target_fps
        self._last_time = time.time()

        self.connect()

    def connect(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        # 读取一帧获取分辨率
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Video read failed at start.")

        h, w = frame.shape[:2]
        self.im_width = w
        self.im_height = h

        # 重置回第一帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 伪内参（保持字段存在）
        fx = 0.9 * w
        fy = 0.9 * w
        cx = w / 2.0
        cy = h / 2.0
        self.intrinsics = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]], dtype=np.float32)

        print("Video camera connected ...")
        print(f"resolution: {w}x{h}")
        print(f"target fps: {self.target_fps}, playback speed: {self.playback_speed}x")

    def get_data(self):
        """
        模拟实时相机帧率控制
        """

        # 控制算法侧“看到的帧率”
        now = time.time()
        dt = now - self._last_time
        if dt < self._frame_interval:
            time.sleep(self._frame_interval - dt)
        self._last_time = time.time()

        # 根据播放倍率跳帧
        skip = int(max(self.playback_speed - 1, 0))
        for _ in range(skip):
            self.cap.grab()

        ok, color_image = self.cap.read()

        # 视频结束 → 循环播放
        if not ok:
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # ok, color_image = self.cap.read()
            # if not ok:
                return None, None

        # 构造假深度
        depth_image = np.full((self.im_height, self.im_width, 1),
                              float(self.default_depth),
                              dtype=np.float32)

        return color_image, depth_image

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


if __name__ == "__main__":


    # cam = USBCamera(width=640, height=480, fps=30, default_depth=0.50, device_index=2)
    # while True:
    #     color, depth = cam.get_data()
    #     if color is None:
    #         continue
    #     cv2.imshow("usb_color", color)
    #     if cv2.waitKey(1) & 0xFF == 27:  # ESC
    #         break
    # cam.release()
    # cv2.destroyAllWindows()

    camera = FakeCamera(
        video_path="videos/tracking_20260127_173847.mp4",
        fps=30,             # 算法处理帧率
        playback_speed=1.0, # 视频 2 倍速播放
        default_depth=0.4
    )

    while True:
        color, depth = camera.get_data()
        if color is None:
            continue
        cv2.imshow("color", color)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    cv2.destroyAllWindows()


    #     mycamera = RSD435i()
    #     while True:
    #     # color_image, depth_image = mycamera.get_data()

    #         mycamera.plot_image()
    #     # print(mycamera.intrinsics)