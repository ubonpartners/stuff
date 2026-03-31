"""Inference backend framework for `stuff.inference_wrapper`."""

from .types import InferenceConfig, ModelSpec
from .postprocess import (
    CLASS_SYNONYMS,
    build_class_mapping,
    ensemble_merge_results,
    merge_detections_wbf,
)

__all__ = [
    "InferenceConfig",
    "ModelSpec",
    "CLASS_SYNONYMS",
    "build_class_mapping",
    "merge_detections_wbf",
    "ensemble_merge_results",
]
