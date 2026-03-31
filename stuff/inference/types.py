"""Shared types for inference backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceConfig:
    model_name: str | list[str]
    class_names: list[str] | None = None
    thr: float = 0.5
    nms_thr: float = 0.45
    half: bool = True
    rect: bool = True
    max_det: int = 1200
    batch_size: int = 32
    imgsz: int = 640
    attributes: list[str] | None = None
    face_kp: bool = False
    pose_kp: bool = False
    facepose_kp: bool = False
    fold_attributes: bool = False
    get_feats: bool = False


@dataclass
class ModelSpec:
    model_name: str | list[str]
    model_name_list: list[str]
    imgsz: int
    aug_params: dict[str, Any] = field(default_factory=dict)
