"""Unified inference wrapper over multiple detector backends."""

from __future__ import annotations

import os

import requests

from .inference.factory import create_backend, parse_model_spec
from .inference.postprocess import (
    CLASS_SYNONYMS,
    build_class_mapping,
    ensemble_merge_results,
    merge_detections_wbf,
)
from .inference.types import InferenceConfig


def download_mmdet_config(config_path, repo="https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs"):
    """
    Download MMDetection config if it does not exist locally.

    Args:
        config_path (str): Relative path like 'rtmdet/rtmdet-h_8xb32-300e_coco.py'.
        repo (str): Base URL to fetch from.
    Returns:
        str: Path to the saved config.
    """
    config_dir = os.path.join("mmdetection", "configs")
    local_path = os.path.join(config_dir, config_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.isfile(local_path):
        print(f"Downloading MMDetection config: {config_path}")
        url = f"{repo}/{config_path}"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise FileNotFoundError(f"Failed to download config: {url}")
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(response.text)
    return local_path


def github_blob_to_raw_url(blob_url):
    return blob_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")


def infer_model_name(x):
    ext = ""
    if isinstance(x, str) and "," in x:
        x = x.split(",")[0]
    if isinstance(x, str):
        if "+" in x:
            t = x.split("+")
            x = t[0]
            ext += t[1]
        if ":" in x:
            t = x.split(":")
            x = t[0]
            ext += t[1] + " "
        if x.endswith(".engine"):
            ext += "TRT "
        if len(ext) > 0:
            ext = "(" + ext[:-1] + ")"
    return os.path.splitext(os.path.basename(x))[0] + ext


class InferenceWrapper:
    class_synonyms = CLASS_SYNONYMS

    def __init__(
        self,
        model_name,
        class_names=None,
        thr=0.5,
        nms_thr=0.45,
        half=True,
        rect=True,
        max_det=1200,
        batch_size=32,
        imgsz=640,
        attributes=None,
        face_kp=False,
        pose_kp=False,
        facepose_kp=False,
        fold_attributes=False,
        get_feats=False,
    ):
        self.infer_cache = {}
        self.ensemble_models = None
        self.backend = None

        self.config = InferenceConfig(
            model_name=model_name,
            class_names=class_names,
            thr=thr,
            nms_thr=nms_thr,
            half=half,
            rect=rect,
            max_det=max_det,
            batch_size=batch_size,
            imgsz=imgsz,
            attributes=attributes,
            face_kp=face_kp,
            pose_kp=pose_kp,
            facepose_kp=facepose_kp,
            fold_attributes=fold_attributes,
            get_feats=get_feats,
        )

        model_spec = parse_model_spec(model_name, imgsz)
        self.config.imgsz = model_spec.imgsz

        if isinstance(model_spec.model_name, list):
            self.ensemble_models = []
            child_batch_size = max(1, batch_size // 2)
            for child_model_name in model_spec.model_name:
                child = InferenceWrapper(
                    child_model_name,
                    class_names=class_names,
                    thr=thr,
                    nms_thr=nms_thr,
                    max_det=max_det,
                    rect=rect,
                    imgsz=self.config.imgsz,
                    batch_size=child_batch_size,
                    half=half,
                    attributes=attributes,
                    face_kp=face_kp,
                    pose_kp=pose_kp,
                    facepose_kp=facepose_kp,
                    fold_attributes=fold_attributes,
                    get_feats=get_feats,
                )
                self.ensemble_models.append(child)

            self.num_threads = self.ensemble_models[0].num_threads
            self.infer_batch_size = self.ensemble_models[0].infer_batch_size
            self.yolo_num_params = sum(x.yolo_num_params for x in self.ensemble_models)
            self.yolo_num_flops = sum(x.yolo_num_flops for x in self.ensemble_models)
            self.class_names = self.ensemble_models[0].class_names
            self.attributes = self.ensemble_models[0].attributes
            self.infer_model_class_names = self.ensemble_models[0].infer_model_class_names
            self.det_class_remap = self.ensemble_models[0].det_class_remap
            self.can_detect = self.ensemble_models[0].can_detect
            return

        self.backend = create_backend(model_spec, self.config, self.class_synonyms)
        if self.backend is None:
            raise ValueError(f"Unsupported model spec: {model_name}")

        self.infer_model_class_names = self.backend.infer_model_class_names
        self.class_names, self.det_class_remap, self.can_detect = build_class_mapping(
            self.infer_model_class_names, self.config.class_names, class_synonyms=self.class_synonyms
        )
        self.attributes = self.config.attributes
        self.backend.configure_mapping(self.class_names, self.det_class_remap, self.attributes)

        self.num_threads = self.backend.num_threads
        self.infer_batch_size = self.backend.infer_batch_size
        self.yolo_num_params = self.backend.yolo_num_params
        self.yolo_num_flops = self.backend.yolo_num_flops

        # Compatibility aliases used externally in some scripts.
        self.rf_detr_model = getattr(self.backend, "rf_detr_model", None)
        self.detectron2_predictor = getattr(self.backend, "predictor", None)
        self.mmdet_model = (
            getattr(self.backend, "model", None) if self.backend.__class__.__name__ == "MMDetBackend" else None
        )
        self.upyc_infer = getattr(self.backend, "upyc_infer", None)
        self.track_stream = getattr(self.backend, "track_stream", None)

    def set_class_remap(self):
        self.class_names, self.det_class_remap, self.can_detect = build_class_mapping(
            self.infer_model_class_names, self.class_names, class_synonyms=self.class_synonyms
        )
        if self.backend is not None:
            self.backend.configure_mapping(self.class_names, self.det_class_remap, self.attributes)

    def infer(self, input_frames):
        if self.ensemble_models is not None:
            results = []
            for model in self.ensemble_models:
                results.append(model.infer(input_frames))
            return ensemble_merge_results(results)
        return self.backend.infer(input_frames)

    def infer_cached(self, index, num_images, get_image_fn):
        if index in self.infer_cache:
            return self.infer_cache[index]
        self.infer_cache = {}
        num = self.infer_batch_size * self.num_threads
        num = min(num, num_images - index)
        image_list = [get_image_fn(i + index) for i in range(num)]
        batch_results = self.infer(image_list)
        self.infer_cache = {(i + index): batch_results[i] for i in range(num)}
        assert index in self.infer_cache
        return self.infer_cache[index]


inference_wrapper = InferenceWrapper

__all__ = [
    "download_mmdet_config",
    "github_blob_to_raw_url",
    "infer_model_name",
    "merge_detections_wbf",
    "ensemble_merge_results",
    "InferenceWrapper",
    "inference_wrapper",
]