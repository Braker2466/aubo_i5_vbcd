import numpy as np
import pyrealsense2 as rs
import cv2

class RSD435i(object):

    def __init__(self,width=1280,height=720,fps=30,default_depth=None):
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

    def pixel_to_camera(self, x, y, depth_image):
        """
        像素坐标 -> 相机坐标系 (单位: 米)

        Args:
            x, y: 像素坐标 (int 或 float)
            depth_image: 对齐到彩色图的深度图 (H, W, 1)

        Returns:
            np.ndarray (3,): [X, Y, Z] in camera frame
            若坐标越界或深度无效，返回 None
        """

        # -------- 1. 转 int（Kalman 输出常为 float）
        x = int(round(x))
        y = int(round(y))

        # -------- 2. 图像边界检查
        h, w = depth_image.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return None

        # -------- 3. 读取深度（mm -> m）
        z_raw = depth_image[y, x, 0]
        if not np.isfinite(z_raw) or z_raw <= 0.06:
            if self.default_depth is not None:
                z_raw = self.default_depth #深度获取不到/深度小于0.06m的时候，置为default_depth
            else:
                return None

        z = z_raw * self.scale

        # -------- 4. 相机内参
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]

        # -------- 5. 反投影
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy

        return np.array([X, Y, z], dtype=np.float32)


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
if __name__== '__main__':
    mycamera = RSD435i()
    while True:
    # color_image, depth_image = mycamera.get_data()

        mycamera.plot_image()
    # print(mycamera.intrinsics)