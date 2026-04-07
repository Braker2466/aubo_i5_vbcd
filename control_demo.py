from multiprocessing import Process, Queue
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import numpy as np
import time

from core.Aubo_Robot import Aubo_Robot
from UDPReceiver import UDPTrackingReceiver
from TrackingSHMReceiver import TrackingSHMDoubleReceiver


# =========================================================
# Queue helper
# =========================================================
def put_latest(mp_queue, data):
    """
    始终只保留最新一帧
    """
    while True:
        try:
            mp_queue.put_nowait(data)
            break
        except Exception:
            try:
                mp_queue.get_nowait()
            except Exception:
                break


# =========================================================
# Receiver processes
# =========================================================
def UDP_receiver_process(queue):
    receiver = UDPTrackingReceiver(port=9000)
    last_timestamp = None
    print("[UDP Receiver] waiting for data...")
    while True:
        data = receiver.recv_latest()
        if data is None:
            continue

        ts = data.get("timestamp")
        if ts is None:
            continue

        if last_timestamp is not None and ts <= last_timestamp:
            continue

        last_timestamp = ts
        put_latest(queue, data)


def SHM_receiver_process(queue):
    receiver = TrackingSHMDoubleReceiver()
    print("[SHM Receiver] waiting for data...")
    try:
        while True:
            data = receiver.recv_latest()
            if data is None or not data.get("valid", False):
                time.sleep(0.001)
                continue

            put_latest(queue, data)
    except KeyboardInterrupt:
        print("[SHM Receiver] exiting...")
    finally:
        receiver.close()


# =========================================================
# Modes
# =========================================================
class ControlMode(Enum):
    BUILD_V = 0
    FAST_ALIGN = 1
    CHASE = 2
    WAIT_FOLLOW = 3
    FOLLOW = 4
    EXECUTE = 5


# =========================================================
# Config
# =========================================================
@dataclass
class ControllerConfig:
    # ---------- image / center ----------
    image_cx: float = 320.0
    image_cy: float = 240.0

    # ---------- weighted pixel distance ----------
    # d = sqrt(wx * ex^2 + wy * ey^2), x 权重大
    wx_err: float = 1.0
    wy_err: float = 0.25

    # FOLLOW 阶段“相对静止误差”中像素速度的权重
    wx_static_vel: float = 1.0
    wy_static_vel: float = 0.25

    # ---------- center ROI ----------
    roi_half_w: float = 80.0
    roi_half_h: float = 50.0

    # ---------- BUILD_V ----------
    build_v_window_size: int = 20
    build_v_min_samples: int = 8
    build_v_speed_std_max_cam: float = 0.03
    build_v_speed_std_max_pix: float = 50.0
    build_v_dir_cos_min: float = 0.92

    # ---------- FAST_ALIGN ----------
    enable_fast_align_prediction: bool = True
    fast_align_lead_time: float = 0.08
    fast_align_max_stale_time: float = 0.20

    v_fast_align: float = 0.12
    a_fast_align: float = 0.12
    step_fast_align: float = 0.020

    # ---------- CHASE ----------
    # 触发追赶：目标距离中心 <= d_trigger
    d_trigger_init: float = 90.0
    d_trigger_min: float = 8.0
    d_trigger_max: float = 140.0

    step_chase: float = 0.010
    v_chase_init: float = 0.08
    a_chase_init: float = 0.08

    # 保证“追上并超过”
    chase_speed_gain: float = 1.20
    chase_speed_margin: float = 0.01
    v_chase_inc: float = 0.01
    a_chase_inc: float = 0.01
    v_chase_max: float = 0.30
    a_chase_max: float = 0.30

    # ---------- WAIT_FOLLOW ----------
    follow_delay_ms: float = 120.0

    # ---------- FOLLOW ----------
    step_follow: float = 0.006
    v_follow_init_gain: float = 1.00
    a_follow_fixed: float = 0.05

    v_follow_inc: float = 0.003
    v_follow_min: float = 0.01
    v_follow_max: float = 0.20

    # 伴随误差调节参数
    follow_drift_deadband_pix_s: float = 2.0

    # ---------- static evaluation ----------
    # 仅评估阻塞段中的匀速部分
    uniform_phase_start_ratio: float = 0.20
    uniform_phase_end_ratio: float = 0.80

    static_speed_threshold: float = 6.0      # px/s
    static_ratio_accept: float = 0.80
    e_static_accept: float = 8.0             # 平均加权像素速度
    d2_accept: float = 10.0                  # 相对静止时的中心残余偏差
    follow_success_needed: int = 2

    # d_trigger <- d_trigger - k * d2
    d2_feedback_gain: float = 0.12

    # ---------- monitor ----------
    monitor_queue_poll_dt: float = 0.002

    # ---------- tool / motion ----------
    z_fixed: float = 0.50
    use_fixed_z: bool = True
    xy_deadband: float = 0.001

    # ---------- misc ----------
    verbose: bool = True


# =========================================================
# Runtime params
# =========================================================
@dataclass
class RuntimeParams:
    d_trigger: float
    v_chase: float
    a_chase: float
    v_follow: float


# =========================================================
# Observation
# =========================================================
@dataclass
class Observation:
    timestamp: float
    p_cam: np.ndarray
    v_cam: np.ndarray
    pixel: np.ndarray | None
    raw: dict


# =========================================================
# Segment record
# =========================================================
@dataclass
class SegmentRecord:
    mode: ControlMode
    start_time: float
    end_time: float | None = None
    duration: float | None = None

    # 原始误差序列
    err_series: list = field(default_factory=list)          # weighted distance
    ex_series: list = field(default_factory=list)
    ey_series: list = field(default_factory=list)
    de_series: list = field(default_factory=list)

    # 像素运动相关
    du_dt_series: list = field(default_factory=list)
    dv_dt_series: list = field(default_factory=list)
    pixel_speed_series: list = field(default_factory=list)  # weighted pixel speed
    signed_drift_series: list = field(default_factory=list) # 沿主方向漂移速度

    # 时间序列
    ts_series: list = field(default_factory=list)

    n_samples: int = 0

    # 段整体统计
    start_err: float | None = None
    end_err: float | None = None
    mean_err: float | None = None
    var_err: float | None = None
    mean_de: float | None = None

    # FOLLOW / 匀速阶段统计
    uniform_sample_count: int = 0
    static_ratio: float | None = None
    e_static: float | None = None
    e_center: float | None = None
    d2: float | None = None
    mean_signed_drift: float | None = None

    # 诊断
    note: str = ""


# =========================================================
# Utilities
# =========================================================
def norm2(x):
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def safe_normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


# =========================================================
# Shared latest observation buffer
# =========================================================
class LatestObservationBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Observation | None = None

    def set(self, obs: Observation):
        with self._lock:
            self._latest = obs

    def get(self) -> Observation | None:
        with self._lock:
            return self._latest


# =========================================================
# Segment monitor
# =========================================================
class SegmentMonitor:
    """
    在机械臂阻塞运动期间持续读取最新视觉数据，只做监视分析，不下发新命令。
    支持 FOLLOW 阶段的“匀速段相对静止”评估。
    """
    def __init__(self, queue, obs_buffer: LatestObservationBuffer, cfg: ControllerConfig):
        self.queue = queue
        self.obs_buffer = obs_buffer
        self.cfg = cfg

        self._active = False
        self._thread = None
        self._record: SegmentRecord | None = None

        self._prev_obs: Observation | None = None
        self._signed_drift_axis = np.array([1.0, 0.0], dtype=float)

    def _parse_data(self, data) -> Observation | None:
        if data is None:
            return None

        cam = data.get("camera", None)
        pix = data.get("pixel", None)
        ts = data.get("timestamp", time.time())

        if cam is None:
            return None

        cam = np.asarray(cam, dtype=float).reshape(-1)
        if cam.size < 3:
            return None

        p_cam = cam[:3].copy()
        v_cam = cam[3:6].copy() if cam.size >= 6 else np.zeros(3, dtype=float)

        pixel = None
        if pix is not None:
            pix = np.asarray(pix, dtype=float).reshape(-1)
            if pix.size >= 2:
                pixel = pix[:2].copy()

        return Observation(
            timestamp=float(ts),
            p_cam=p_cam,
            v_cam=v_cam,
            pixel=pixel,
            raw=data
        )

    def _weighted_distance(self, ex: float, ey: float) -> float:
        return float(np.sqrt(
            self.cfg.wx_err * ex * ex +
            self.cfg.wy_err * ey * ey
        ))

    def _weighted_pixel_speed(self, du_dt: float, dv_dt: float) -> float:
        return float(np.sqrt(
            self.cfg.wx_static_vel * du_dt * du_dt +
            self.cfg.wy_static_vel * dv_dt * dv_dt
        ))

    def _compute_errors(self, obs: Observation):
        if obs.pixel is None:
            return None

        ex = float(obs.pixel[0] - self.cfg.image_cx)
        ey = float(obs.pixel[1] - self.cfg.image_cy)
        err = self._weighted_distance(ex, ey)
        return ex, ey, err

    def _worker(self):
        while self._active:
            latest_data = None
            try:
                while True:
                    latest_data = self.queue.get_nowait()
            except Exception:
                pass

            if latest_data is not None:
                obs = self._parse_data(latest_data)
                if obs is not None:
                    self.obs_buffer.set(obs)
                    self._append(obs)

            time.sleep(self.cfg.monitor_queue_poll_dt)

    def _append(self, obs: Observation):
        if self._record is None:
            return

        errs = self._compute_errors(obs)
        if errs is None:
            return

        ex, ey, err = errs

        if self._record.start_err is None:
            self._record.start_err = err

        self._record.err_series.append(err)
        self._record.ex_series.append(ex)
        self._record.ey_series.append(ey)
        self._record.ts_series.append(obs.timestamp)
        self._record.n_samples += 1

        if self._prev_obs is not None and self._prev_obs.pixel is not None:
            dt = max(1e-6, obs.timestamp - self._prev_obs.timestamp)
            prev_ex = float(self._prev_obs.pixel[0] - self.cfg.image_cx)
            prev_ey = float(self._prev_obs.pixel[1] - self.cfg.image_cy)
            prev_err = self._weighted_distance(prev_ex, prev_ey)

            de = (err - prev_err) / dt
            du_dt = float((obs.pixel[0] - self._prev_obs.pixel[0]) / dt)
            dv_dt = float((obs.pixel[1] - self._prev_obs.pixel[1]) / dt)
            pix_speed = self._weighted_pixel_speed(du_dt, dv_dt)

            signed_drift = float(np.dot(
                np.array([du_dt, dv_dt], dtype=float),
                self._signed_drift_axis
            ))
        else:
            de = 0.0
            du_dt = 0.0
            dv_dt = 0.0
            pix_speed = 0.0
            signed_drift = 0.0

        self._record.de_series.append(float(de))
        self._record.du_dt_series.append(float(du_dt))
        self._record.dv_dt_series.append(float(dv_dt))
        self._record.pixel_speed_series.append(float(pix_speed))
        self._record.signed_drift_series.append(float(signed_drift))

        self._prev_obs = obs

    def start(self, mode: ControlMode, drift_axis_pix=None):
        self._record = SegmentRecord(
            mode=mode,
            start_time=time.time()
        )
        self._prev_obs = None

        if drift_axis_pix is not None:
            axis = np.asarray(drift_axis_pix, dtype=float).reshape(-1)
            if axis.size >= 2 and np.linalg.norm(axis[:2]) > 1e-9:
                self._signed_drift_axis = safe_normalize(axis[:2])
            else:
                self._signed_drift_axis = np.array([1.0, 0.0], dtype=float)
        else:
            self._signed_drift_axis = np.array([1.0, 0.0], dtype=float)

        self._active = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> SegmentRecord:
        self._active = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if self._record is None:
            raise RuntimeError("SegmentMonitor stop called before start.")

        self._record.end_time = time.time()
        self._record.duration = self._record.end_time - self._record.start_time

        self._finalize_record(self._record)
        return self._record

    def _finalize_record(self, rec: SegmentRecord):
        if len(rec.err_series) > 0:
            rec.end_err = float(rec.err_series[-1])
            rec.mean_err = float(np.mean(rec.err_series))
            rec.var_err = float(np.var(rec.err_series))
            rec.mean_de = float(np.mean(rec.de_series)) if len(rec.de_series) > 0 else 0.0
        else:
            rec.end_err = rec.start_err
            rec.mean_err = None
            rec.var_err = None
            rec.mean_de = None

        if rec.mode == ControlMode.FOLLOW:
            self._finalize_follow_metrics(rec)

    def _finalize_follow_metrics(self, rec: SegmentRecord):
        n = rec.n_samples
        if n < 3:
            rec.note = "FOLLOW samples too few"
            rec.static_ratio = 0.0
            rec.e_static = None
            rec.e_center = None
            rec.d2 = None
            rec.mean_signed_drift = None
            return

        start_idx = int(np.floor(self.cfg.uniform_phase_start_ratio * n))
        end_idx = int(np.ceil(self.cfg.uniform_phase_end_ratio * n))

        start_idx = clamp(start_idx, 0, n - 1)
        end_idx = clamp(end_idx, start_idx + 1, n)

        uniform_pixel_speed = rec.pixel_speed_series[start_idx:end_idx]
        uniform_err = rec.err_series[start_idx:end_idx]
        uniform_signed_drift = rec.signed_drift_series[start_idx:end_idx]

        rec.uniform_sample_count = len(uniform_pixel_speed)

        if rec.uniform_sample_count <= 0:
            rec.note = "FOLLOW uniform phase empty"
            rec.static_ratio = 0.0
            rec.e_static = None
            rec.e_center = None
            rec.d2 = None
            rec.mean_signed_drift = None
            return

        static_mask = [1.0 if s <= self.cfg.static_speed_threshold else 0.0
                       for s in uniform_pixel_speed]
        rec.static_ratio = float(np.mean(static_mask))
        rec.e_static = float(np.mean(uniform_pixel_speed))
        rec.e_center = float(np.mean(uniform_err))
        rec.mean_signed_drift = float(np.mean(uniform_signed_drift))

        # d2：相对静止时的像素中心残余偏差
        # 取匀速段后半段均值，更接近稳定后残差
        tail_start = start_idx + max(1, (end_idx - start_idx) // 2)
        tail_err = rec.err_series[tail_start:end_idx]
        if len(tail_err) > 0:
            rec.d2 = float(np.mean(tail_err))
        else:
            rec.d2 = rec.e_center


# =========================================================
# Controller
# =========================================================
class BlockingVisualServoController:
    def __init__(self, robot: Aubo_Robot, queue, cfg: ControllerConfig):
        self.robot = robot
        self.queue = queue
        self.cfg = cfg

        self.mode = ControlMode.BUILD_V
        self.obs_buffer = LatestObservationBuffer()
        self.monitor = SegmentMonitor(queue, self.obs_buffer, cfg)

        self.params = RuntimeParams(
            d_trigger=cfg.d_trigger_init,
            v_chase=cfg.v_chase_init,
            a_chase=cfg.a_chase_init,
            v_follow=clamp(
                cfg.v_follow_init_gain * cfg.v_chase_init,
                cfg.v_follow_min,
                cfg.v_follow_max
            ),
        )

        self.prev_obs: Observation | None = None

        # BUILD_V 用的窗口
        self.velocity_obs_window = deque(maxlen=cfg.build_v_window_size)

        # 参考速度与主方向
        self.v_ref_cam = np.zeros(3, dtype=float)
        self.v_ref_pix = np.zeros(2, dtype=float)
        self.dir_cam_unit = np.array([1.0, 0.0, 0.0], dtype=float)
        self.dir_pix_unit = np.array([1.0, 0.0], dtype=float)

        self.follow_success_count = 0
        self.chase_end_time = None

        self.history = deque(maxlen=100)

    # -----------------------------------------------------
    # custom interfaces
    # -----------------------------------------------------
    def ready_to_execute(self, rec: SegmentRecord | None) -> bool:
        if rec is None:
            return False

        if rec.mode != ControlMode.FOLLOW:
            return False

        return (
            rec.static_ratio is not None
            and rec.e_static is not None
            and rec.d2 is not None
            and rec.static_ratio >= self.cfg.static_ratio_accept
            and rec.e_static <= self.cfg.e_static_accept
            and rec.d2 <= self.cfg.d2_accept
        )

    def execute_task_blocking(self):
        """
        EXECUTE 阶段占位函数。
        你后续可替换为实际动作。
        """
        print("[EXECUTE] placeholder task...")
        time.sleep(0.5)

    # -----------------------------------------------------
    # parsing
    # -----------------------------------------------------
    def parse_data(self, data) -> Observation | None:
        if data is None:
            return None

        cam = data.get("camera", None)
        pix = data.get("pixel", None)
        ts = data.get("timestamp", time.time())

        if cam is None:
            return None

        cam = np.asarray(cam, dtype=float).reshape(-1)
        if cam.size < 3:
            return None

        p_cam = cam[:3].copy()
        v_cam = cam[3:6].copy() if cam.size >= 6 else np.zeros(3, dtype=float)

        pixel = None
        if pix is not None:
            pix = np.asarray(pix, dtype=float).reshape(-1)
            if pix.size >= 2:
                pixel = pix[:2].copy()

        return Observation(
            timestamp=float(ts),
            p_cam=p_cam,
            v_cam=v_cam,
            pixel=pixel,
            raw=data
        )

    # -----------------------------------------------------
    # geometry / metrics
    # -----------------------------------------------------
    def compute_error_components(self, obs: Observation):
        if obs.pixel is None:
            return None

        ex = float(obs.pixel[0] - self.cfg.image_cx)
        ey = float(obs.pixel[1] - self.cfg.image_cy)
        d = float(np.sqrt(
            self.cfg.wx_err * ex * ex +
            self.cfg.wy_err * ey * ey
        ))
        return ex, ey, d

    def is_inside_center_roi(self, obs: Observation) -> bool:
        errs = self.compute_error_components(obs)
        if errs is None:
            return False
        ex, ey, _ = errs
        return (abs(ex) <= self.cfg.roi_half_w) and (abs(ey) <= self.cfg.roi_half_h)

    def compute_pixel_velocity(self, obs_curr: Observation, obs_prev: Observation):
        if obs_curr.pixel is None or obs_prev.pixel is None:
            return np.zeros(2, dtype=float)

        dt = max(1e-6, obs_curr.timestamp - obs_prev.timestamp)
        return (obs_curr.pixel - obs_prev.pixel) / dt

    # -----------------------------------------------------
    # BUILD_V
    # -----------------------------------------------------
    def append_velocity_window(self, obs: Observation):
        self.velocity_obs_window.append(obs)

    def is_velocity_estimation_stable(self) -> bool:
        if len(self.velocity_obs_window) < self.cfg.build_v_min_samples:
            return False

        # 相机速度稳定性
        v_cam_samples = np.array([o.v_cam for o in self.velocity_obs_window], dtype=float)
        cam_speed_norms = np.linalg.norm(v_cam_samples, axis=1)

        if np.std(cam_speed_norms) > self.cfg.build_v_speed_std_max_cam:
            return False

        # 像素速度稳定性
        pix_vels = []
        for i in range(1, len(self.velocity_obs_window)):
            o0 = self.velocity_obs_window[i - 1]
            o1 = self.velocity_obs_window[i]
            if o0.pixel is None or o1.pixel is None:
                continue
            pix_vels.append(self.compute_pixel_velocity(o1, o0))

        if len(pix_vels) < max(3, self.cfg.build_v_min_samples - 1):
            return False

        pix_vels = np.array(pix_vels, dtype=float)
        pix_speed_norms = np.linalg.norm(pix_vels, axis=1)
        if np.std(pix_speed_norms) > self.cfg.build_v_speed_std_max_pix:
            return False

        mean_pix_dir = safe_normalize(np.mean(pix_vels, axis=0))
        cos_vals = []
        for v in pix_vels:
            vn = safe_normalize(v)
            if np.linalg.norm(vn) < 1e-9 or np.linalg.norm(mean_pix_dir) < 1e-9:
                continue
            cos_vals.append(float(np.dot(vn, mean_pix_dir)))

        if len(cos_vals) == 0:
            return False

        if float(np.mean(cos_vals)) < self.cfg.build_v_dir_cos_min:
            return False

        return True

    def estimate_reference_velocity(self):
        if len(self.velocity_obs_window) < 2:
            return

        v_cam_samples = np.array([o.v_cam for o in self.velocity_obs_window], dtype=float)
        self.v_ref_cam = np.mean(v_cam_samples, axis=0)

        pix_vels = []
        for i in range(1, len(self.velocity_obs_window)):
            o0 = self.velocity_obs_window[i - 1]
            o1 = self.velocity_obs_window[i]
            if o0.pixel is None or o1.pixel is None:
                continue
            pix_vels.append(self.compute_pixel_velocity(o1, o0))

        if len(pix_vels) > 0:
            pix_vels = np.array(pix_vels, dtype=float)
            self.v_ref_pix = np.mean(pix_vels, axis=0)
        else:
            self.v_ref_pix = np.zeros(2, dtype=float)

        self.dir_cam_unit = safe_normalize(self.v_ref_cam)
        if np.linalg.norm(self.dir_cam_unit) < 1e-9:
            self.dir_cam_unit = np.array([1.0, 0.0, 0.0], dtype=float)

        self.dir_pix_unit = safe_normalize(self.v_ref_pix)
        if np.linalg.norm(self.dir_pix_unit) < 1e-9:
            self.dir_pix_unit = np.array([1.0, 0.0], dtype=float)

        # 初始化 FOLLOW 速度
        v_target = norm2(self.v_ref_cam)
        self.params.v_follow = clamp(
            self.cfg.v_follow_init_gain * v_target,
            self.cfg.v_follow_min,
            self.cfg.v_follow_max
        )

        if self.cfg.verbose:
            print(
                "[BUILD_V] "
                f"v_ref_cam={np.round(self.v_ref_cam, 4)}, "
                f"v_ref_pix={np.round(self.v_ref_pix, 4)}, "
                f"dir_cam={np.round(self.dir_cam_unit, 4)}, "
                f"dir_pix={np.round(self.dir_pix_unit, 4)}, "
                f"v_follow_init={self.params.v_follow:.4f}"
            )

    # -----------------------------------------------------
    # FAST_ALIGN
    # -----------------------------------------------------
    def compensate_fast_align_target(self, obs: Observation):
        if not self.cfg.enable_fast_align_prediction:
            return obs.p_cam.copy()

        t_now = time.time()
        stale = max(0.0, t_now - obs.timestamp)
        stale = min(stale, self.cfg.fast_align_max_stale_time)
        tau = stale + self.cfg.fast_align_lead_time
        return obs.p_cam + tau * self.v_ref_cam

    def execute_fast_align(self, obs: Observation):
        p_cmd_cam = self.compensate_fast_align_target(obs)
        target_pos_base = self.robot.compute_target2base(p_cmd_cam)

        self.robot.set_end_max_line_velc(self.cfg.v_fast_align)
        self.robot.set_end_max_line_acc(self.cfg.a_fast_align)
        self.robot.set_joint_maxacc((1, 1, 1, 1, 1, 1))
        self.robot.set_joint_maxvelc((1, 1, 1, 1, 1, 1))

        if self.cfg.verbose:
            errs = self.compute_error_components(obs)
            d_now = errs[2] if errs is not None else None
            print(
                "[FAST_ALIGN] "
                f"d_now={d_now}, "
                f"v={self.cfg.v_fast_align:.3f}, "
                f"a={self.cfg.a_fast_align:.3f}, "
                f"step={self.cfg.step_fast_align:.4f}, "
                f"p_obs={np.round(obs.p_cam, 4)}, "
                f"p_cmd={np.round(p_cmd_cam, 4)}, "
                f"target_base={np.round(target_pos_base, 4)}"
            )

        self.monitor.start(ControlMode.FAST_ALIGN, drift_axis_pix=self.dir_pix_unit)

        ok = self.robot.align_to_target_line_stepwise(
            position=target_pos_base,
            z_offset=self.cfg.z_fixed,
            use_fixed_z=self.cfg.use_fixed_z,
            max_step_xy=self.cfg.step_fast_align,
            xy_deadband=self.cfg.xy_deadband
        )

        rec = self.monitor.stop()
        return ok, rec

    # -----------------------------------------------------
    # CHASE
    # -----------------------------------------------------
    def ensure_chase_speed_capability(self):
        target_speed = norm2(self.v_ref_cam)
        required = self.cfg.chase_speed_gain * target_speed + self.cfg.chase_speed_margin

        while self.params.v_chase < required and self.params.v_chase < self.cfg.v_chase_max:
            self.params.v_chase = clamp(
                self.params.v_chase + self.cfg.v_chase_inc,
                0.0,
                self.cfg.v_chase_max
            )

        # 非必要不调加速度；仅当速度拉高后，加速度仍显著偏低时再补
        # 这里给一个简单保守策略：a_chase 至少不低于 v_chase
        if self.params.a_chase < self.params.v_chase:
            self.params.a_chase = clamp(
                self.params.a_chase + self.cfg.a_chase_inc,
                0.0,
                self.cfg.a_chase_max
            )

        if self.cfg.verbose:
            print(
                "[CHASE_CAP] "
                f"target_speed={target_speed:.4f}, "
                f"required={required:.4f}, "
                f"v_chase={self.params.v_chase:.4f}, "
                f"a_chase={self.params.a_chase:.4f}"
            )

    def execute_chase_segment(self, obs: Observation):
        # 不使用未来预测点，按当前方向固定步长前进
        p_cmd_cam = obs.p_cam + self.dir_cam_unit * self.cfg.step_chase
        target_pos_base = self.robot.compute_target2base(p_cmd_cam)

        self.robot.set_end_max_line_velc(self.params.v_chase)
        self.robot.set_end_max_line_acc(self.params.a_chase)
        self.robot.set_joint_maxacc((1, 1, 1, 1, 1, 1))
        self.robot.set_joint_maxvelc((1, 1, 1, 1, 1, 1))

        if self.cfg.verbose:
            errs = self.compute_error_components(obs)
            d_now = errs[2] if errs is not None else None
            print(
                "[CHASE] "
                f"d_now={d_now}, "
                f"d_trigger={self.params.d_trigger:.2f}, "
                f"v={self.params.v_chase:.3f}, "
                f"a={self.params.a_chase:.3f}, "
                f"step={self.cfg.step_chase:.4f}, "
                f"dir_cam={np.round(self.dir_cam_unit, 4)}, "
                f"p_obs={np.round(obs.p_cam, 4)}, "
                f"p_cmd={np.round(p_cmd_cam, 4)}, "
                f"target_base={np.round(target_pos_base, 4)}"
            )

        self.monitor.start(ControlMode.CHASE, drift_axis_pix=self.dir_pix_unit)

        ok = self.robot.align_to_target_line_stepwise(
            position=target_pos_base,
            z_offset=self.cfg.z_fixed,
            use_fixed_z=self.cfg.use_fixed_z,
            max_step_xy=self.cfg.step_chase,
            xy_deadband=self.cfg.xy_deadband
        )

        rec = self.monitor.stop()
        return ok, rec

    # -----------------------------------------------------
    # WAIT_FOLLOW
    # -----------------------------------------------------
    def wait_follow_ready(self):
        if self.chase_end_time is None:
            return True
        elapsed_ms = (time.time() - self.chase_end_time) * 1000.0
        return elapsed_ms >= self.cfg.follow_delay_ms

    # -----------------------------------------------------
    # FOLLOW
    # -----------------------------------------------------
    def execute_follow_segment(self, obs: Observation):
        # 不做未来预测，固定步长沿当前主方向伴随
        p_cmd_cam = obs.p_cam + self.dir_cam_unit * self.cfg.step_follow
        target_pos_base = self.robot.compute_target2base(p_cmd_cam)

        self.robot.set_end_max_line_velc(self.params.v_follow)
        self.robot.set_end_max_line_acc(self.cfg.a_follow_fixed)
        self.robot.set_joint_maxacc((1, 1, 1, 1, 1, 1))
        self.robot.set_joint_maxvelc((1, 1, 1, 1, 1, 1))

        if self.cfg.verbose:
            errs = self.compute_error_components(obs)
            d_now = errs[2] if errs is not None else None
            print(
                "[FOLLOW] "
                f"d_now={d_now}, "
                f"v={self.params.v_follow:.3f}, "
                f"a={self.cfg.a_follow_fixed:.3f}, "
                f"step={self.cfg.step_follow:.4f}, "
                f"dir_cam={np.round(self.dir_cam_unit, 4)}, "
                f"dir_pix={np.round(self.dir_pix_unit, 4)}, "
                f"p_obs={np.round(obs.p_cam, 4)}, "
                f"p_cmd={np.round(p_cmd_cam, 4)}, "
                f"target_base={np.round(target_pos_base, 4)}"
            )

        self.monitor.start(ControlMode.FOLLOW, drift_axis_pix=self.dir_pix_unit)

        ok = self.robot.align_to_target_line_stepwise(
            position=target_pos_base,
            z_offset=self.cfg.z_fixed,
            use_fixed_z=self.cfg.use_fixed_z,
            max_step_xy=self.cfg.step_follow,
            xy_deadband=self.cfg.xy_deadband
        )

        rec = self.monitor.stop()
        return ok, rec

    def adapt_follow_speed_from_record(self, rec: SegmentRecord):
        if rec.mode != ControlMode.FOLLOW:
            return

        if rec.mean_signed_drift is None:
            return

        drift = rec.mean_signed_drift

        if abs(drift) <= self.cfg.follow_drift_deadband_pix_s:
            if self.cfg.verbose:
                print(f"[FOLLOW_ADAPT] drift={drift:.3f}, within deadband.")
            return

        # 约定：
        # drift > 0: 目标在图像中沿主方向继续正向漂移，说明机械臂伴随偏慢 -> 增大 v_follow
        # drift < 0: 目标反向漂移，说明机械臂伴随偏快 -> 减小 v_follow
        if drift > 0.0:
            self.params.v_follow += self.cfg.v_follow_inc
        else:
            self.params.v_follow -= self.cfg.v_follow_inc

        self.params.v_follow = clamp(
            self.params.v_follow,
            self.cfg.v_follow_min,
            self.cfg.v_follow_max
        )

        if self.cfg.verbose:
            print(
                "[FOLLOW_ADAPT] "
                f"drift={drift:.3f}, "
                f"v_follow={self.params.v_follow:.4f}"
            )

    # -----------------------------------------------------
    # d_trigger update
    # -----------------------------------------------------
    def update_d_trigger_from_d2(self, d2: float | None):
        if d2 is None:
            return

        correction = self.cfg.d2_feedback_gain * d2
        self.params.d_trigger = clamp(
            self.params.d_trigger - correction,
            self.cfg.d_trigger_min,
            self.cfg.d_trigger_max
        )

        if self.cfg.verbose:
            print(
                "[D_TRIGGER_ADAPT] "
                f"d2={d2:.3f}, "
                f"correction={correction:.3f}, "
                f"d_trigger={self.params.d_trigger:.3f}"
            )

    # -----------------------------------------------------
    # reset
    # -----------------------------------------------------
    def reset_after_execute(self):
        self.mode = ControlMode.BUILD_V
        self.follow_success_count = 0
        self.chase_end_time = None
        self.velocity_obs_window.clear()

        self.v_ref_cam = np.zeros(3, dtype=float)
        self.v_ref_pix = np.zeros(2, dtype=float)
        self.dir_cam_unit = np.array([1.0, 0.0, 0.0], dtype=float)
        self.dir_pix_unit = np.array([1.0, 0.0], dtype=float)

        self.params.v_chase = self.cfg.v_chase_init
        self.params.a_chase = self.cfg.a_chase_init
        self.params.v_follow = clamp(
            self.cfg.v_follow_init_gain * self.cfg.v_chase_init,
            self.cfg.v_follow_min,
            self.cfg.v_follow_max
        )

    # -----------------------------------------------------
    # one blocking step
    # -----------------------------------------------------
    def step(self, latest_data):
        obs = self.parse_data(latest_data)
        if obs is None:
            if self.cfg.verbose:
                print("[CTRL] invalid latest data, skip.")
            return False

        self.obs_buffer.set(obs)

        errs = self.compute_error_components(obs)
        if errs is not None and self.cfg.verbose:
            ex, ey, d_now = errs
            print(
                f"[OBS] mode={self.mode.name}, "
                f"ex={ex:.2f}, ey={ey:.2f}, d={d_now:.2f}"
            )

        # -------------------------------------------------
        # BUILD_V
        # -------------------------------------------------
        if self.mode == ControlMode.BUILD_V:
            self.append_velocity_window(obs)

            if not self.is_velocity_estimation_stable():
                if self.cfg.verbose:
                    print(f"[BUILD_V] collecting... n={len(self.velocity_obs_window)}")
                return False

            self.estimate_reference_velocity()

            if not self.is_inside_center_roi(obs):
                self.mode = ControlMode.FAST_ALIGN
            else:
                self.mode = ControlMode.CHASE

            return True

        # -------------------------------------------------
        # FAST_ALIGN
        # -------------------------------------------------
        if self.mode == ControlMode.FAST_ALIGN:
            if self.is_inside_center_roi(obs):
                if self.cfg.verbose:
                    print("[FAST_ALIGN] target already inside ROI, switch to CHASE.")
                self.mode = ControlMode.CHASE
                return True

            ok, rec = self.execute_fast_align(obs)
            self.history.append(rec)

            latest_obs = self.obs_buffer.get()
            if latest_obs is not None and self.is_inside_center_roi(latest_obs):
                self.mode = ControlMode.CHASE
            else:
                self.mode = ControlMode.FAST_ALIGN

            return ok

        # -------------------------------------------------
        # CHASE
        # -------------------------------------------------
        if self.mode == ControlMode.CHASE:
            if errs is None:
                return False

            _, _, d_now = errs

            # 到达追赶触发距离才开始追赶
            if d_now > self.params.d_trigger:
                if self.cfg.verbose:
                    print(
                        "[CHASE] waiting trigger... "
                        f"d_now={d_now:.2f} > d_trigger={self.params.d_trigger:.2f}"
                    )
                return False

            self.ensure_chase_speed_capability()
            ok, rec = self.execute_chase_segment(obs)
            self.history.append(rec)

            self.chase_end_time = time.time()
            self.mode = ControlMode.WAIT_FOLLOW
            return ok

        # -------------------------------------------------
        # WAIT_FOLLOW
        # -------------------------------------------------
        if self.mode == ControlMode.WAIT_FOLLOW:
            if not self.wait_follow_ready():
                if self.cfg.verbose:
                    elapsed_ms = (time.time() - self.chase_end_time) * 1000.0
                    print(
                        "[WAIT_FOLLOW] "
                        f"elapsed={elapsed_ms:.1f} ms / {self.cfg.follow_delay_ms:.1f} ms"
                    )
                return False

            self.mode = ControlMode.FOLLOW
            return True

        # -------------------------------------------------
        # FOLLOW
        # -------------------------------------------------
        if self.mode == ControlMode.FOLLOW:
            ok, rec = self.execute_follow_segment(obs)
            self.history.append(rec)

            self.adapt_follow_speed_from_record(rec)

            follow_ok = self.ready_to_execute(rec)
            if follow_ok:
                self.follow_success_count += 1
                self.update_d_trigger_from_d2(rec.d2)
            else:
                self.follow_success_count = 0

            if self.cfg.verbose:
                print(
                    "[FOLLOW_EVAL] "
                    f"static_ratio={rec.static_ratio}, "
                    f"e_static={rec.e_static}, "
                    f"d2={rec.d2}, "
                    f"follow_success_count={self.follow_success_count}"
                )

            if self.follow_success_count >= self.cfg.follow_success_needed:
                self.mode = ControlMode.EXECUTE
            else:
                self.mode = ControlMode.CHASE

            return ok

        # -------------------------------------------------
        # EXECUTE
        # -------------------------------------------------
        if self.mode == ControlMode.EXECUTE:
            if self.cfg.verbose:
                print("[CTRL] enter EXECUTE")
            self.execute_task_blocking()
            self.reset_after_execute()
            return True

        return False


# =========================================================
# Control process
# =========================================================
def control_process(queue):
    Aubo_Robot.initialize()
    robot = Aubo_Robot()
    robot.go_home()
    robot.set_param()

    cfg = ControllerConfig(
        image_cx=320.0,
        image_cy=240.0,
        wx_err=1.0,
        wy_err=0.25,
        verbose=True
    )

    controller = BlockingVisualServoController(robot, queue, cfg)

    while True:
        latest_data = None
        try:
            while True:
                latest_data = queue.get_nowait()
        except Exception:
            pass

        if latest_data is None:
            time.sleep(0.005)
            continue

        controller.step(latest_data)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    queue = Queue(maxsize=1)

    # p_recv = Process(target=UDP_receiver_process, args=(queue,))
    p_recv = Process(target=SHM_receiver_process, args=(queue,))
    p_ctrl = Process(target=control_process, args=(queue,))

    p_recv.start()
    p_ctrl.start()

    p_recv.join()
    p_ctrl.join()