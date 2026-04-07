from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_CALIBRATION_PATH = PROJECT_ROOT / "calib" / "config" / "system_calibration.json"
LEGACY_PATHS = {
    "camera_pose": PROJECT_ROOT / "camera_pose.txt",
    "depth_scale": PROJECT_ROOT / "camera_depth_scale.txt",
    "tool_pose": PROJECT_ROOT / "tool_calibration_result.txt",
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json() -> dict[str, Any]:
    if not SYSTEM_CALIBRATION_PATH.exists():
        return {}
    with open(SYSTEM_CALIBRATION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"系统标定配置格式错误: {SYSTEM_CALIBRATION_PATH}")
    return data


def _write_json(data: dict[str, Any]) -> None:
    _ensure_parent(SYSTEM_CALIBRATION_PATH)
    with open(SYSTEM_CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _matrix_from_json(data: dict[str, Any], key: str) -> np.ndarray | None:
    section = data.get(key)
    if not isinstance(section, dict):
        return None
    matrix = section.get("matrix")
    if matrix is None:
        return None
    return np.array(matrix, dtype=np.float64)


def _scalar_from_json(data: dict[str, Any], key: str) -> float | None:
    section = data.get(key)
    if not isinstance(section, dict):
        return None
    value = section.get("value")
    if value is None:
        return None
    return float(value)


def _load_legacy_matrix(key: str) -> np.ndarray:
    return np.loadtxt(LEGACY_PATHS[key], delimiter=" ")


def _load_legacy_scalar(key: str) -> float:
    return float(np.loadtxt(LEGACY_PATHS[key], delimiter=" "))


def load_camera_pose() -> np.ndarray:
    data = _read_json()
    camera_pose = _matrix_from_json(data, "camera_pose")
    if camera_pose is not None:
        return camera_pose
    return _load_legacy_matrix("camera_pose")


def load_depth_scale() -> float:
    data = _read_json()
    depth_scale = _scalar_from_json(data, "depth_scale")
    if depth_scale is not None:
        return depth_scale
    return _load_legacy_scalar("depth_scale")


def load_tool_pose() -> np.ndarray:
    data = _read_json()
    tool_pose = _matrix_from_json(data, "tool_pose")
    if tool_pose is not None:
        return tool_pose
    return _load_legacy_matrix("tool_pose")


def update_system_calibration(
    *,
    camera_pose: np.ndarray | None = None,
    depth_scale: float | None = None,
    tool_pose: np.ndarray | None = None,
) -> dict[str, Any]:
    data = _read_json()
    if not data:
        data = {"version": 1}

    if camera_pose is not None:
        data["camera_pose"] = {
            "matrix": np.asarray(camera_pose, dtype=np.float64).tolist(),
        }

    if depth_scale is not None:
        data["depth_scale"] = {
            "value": float(depth_scale),
        }

    if tool_pose is not None:
        data["tool_pose"] = {
            "matrix": np.asarray(tool_pose, dtype=np.float64).tolist(),
        }

    _write_json(data)
    return data
