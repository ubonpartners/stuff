"""Base class for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    def __init__(self):
        self.infer_model_class_names = []
        self.det_class_remap = []
        self.class_names = None
        self.attributes = None
        self.num_threads = 1
        self.infer_batch_size = 1
        self.yolo_num_params = 0
        self.yolo_num_flops = 0

    def configure_mapping(self, class_names, det_class_remap, attributes):
        self.class_names = class_names
        self.det_class_remap = det_class_remap
        self.attributes = attributes

    @abstractmethod
    def infer(self, input_frames):
        raise NotImplementedError
