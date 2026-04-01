"""Factory for selecting and building inference backends."""

from __future__ import annotations

from .backends import (
    Detectron2Backend,
    MMDetBackend,
    RFDetrBackend,
    UPYCTrackBackend,
    UPYCTrtBackend,
    UltralyticsBackend,
)
from .types import ModelSpec


RF_DETR_MODELS = {"rfdetr-nano", "rfdetr-small", "rfdetr-medium", "rfdetr-large", "rfdetr-base"}
DETECTRON2_MODELS = {"faster_rcnn_X_101_32x8d_FPN_3x", "faster_rcnn_R_101_FPN_3x"}
MMDET_MODELS = {
    "cascade-mask-rcnn_x101-64x4d_fpn_ms-3x",
    "grounding-dino-b",
    "dino-5scale",
    "crowddet2",
}


def parse_model_spec(model_name, imgsz) -> ModelSpec:
    aug_params = {}
    parsed_name = model_name

    if isinstance(parsed_name, str) and ":" in parsed_name:
        base_name, params_text = parsed_name.split(":", 1)
        parsed_name = base_name
        params = params_text.split(",") if "," in params_text else [params_text]
        for param in params:
            if param == "flip-lr":
                aug_params["flip-lr"] = True
            else:
                imgsz = int(param)

    if parsed_name == "ensemble2":
        parsed_name = [
            "/mldata/weights/good/yolo11l.pt:960",
            "/mldata/weights/good/yolo12l.pt:640",
            "/mldata/weights/good/yolov9e.pt:864",
            "/mldata/weights/good/yolov5x6u.pt:1280",
        ]
    if parsed_name == "ensemble3":
        parsed_name = [
            "/mldata/weights/good/yolov9e.pt:512",
            "/mldata/weights/good/yolov9e.pt:640,flip-lr",
            "/mldata/weights/good/yolov9e.pt:864",
            "/mldata/weights/good/yolov9e.pt:960,flip-lr",
        ]

    if isinstance(parsed_name, str) and "," in parsed_name:
        model_name_list = parsed_name.split(",")
        primary_name = model_name_list[0]
    else:
        primary_name = parsed_name
        model_name_list = [parsed_name] if isinstance(parsed_name, str) else []

    return ModelSpec(
        model_name=parsed_name,
        model_name_list=model_name_list,
        imgsz=imgsz,
        aug_params=aug_params,
    )


def create_backend(model_spec, config, class_synonyms):
    model_name = model_spec.model_name
    if isinstance(model_name, list):
        return None

    if model_name in RF_DETR_MODELS:
        return RFDetrBackend(model_name, config)

    if model_name in DETECTRON2_MODELS:
        return Detectron2Backend(model_name, config)

    if model_name in MMDET_MODELS:
        return MMDetBackend(model_name, config)

    if model_name.endswith(".trt"):
        raise ValueError(
            "Direct .trt detector models are no longer supported. "
            "Use a metadata-wrapped .engine file instead."
        )

    if model_name.endswith(".engine"):
        return UPYCTrtBackend(model_name, model_spec.model_name_list, config)

    if model_name == "upyc_track":
        return UPYCTrackBackend(model_spec.model_name_list, config)

    return UltralyticsBackend(model_name, model_spec.aug_params, config, class_synonyms)
