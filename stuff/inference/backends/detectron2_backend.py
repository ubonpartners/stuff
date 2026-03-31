"""Detectron2 backend."""

from __future__ import annotations

import cv2
import numpy as np

from ...image import get_image_size
from ..base import InferenceBackend


class Detectron2Backend(InferenceBackend):
    def __init__(self, model_name, config):
        super().__init__()
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.data import MetadataCatalog
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:
            raise AssertionError("try python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'") from exc

        cfg = get_cfg()
        model = "COCO-Detection/" + model_name + ".yaml"
        cfg.merge_from_file(model_zoo.get_config_file(model))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.thr
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.nms_thr

        self.predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.infer_model_class_names = metadata.get("thing_classes", None)
        self.num_threads = 1
        self.infer_batch_size = config.batch_size

    def infer(self, input_frames):
        ret = []
        for frame in input_frames:
            if not isinstance(frame, np.ndarray):
                image = cv2.imread(frame)
            else:
                image = frame
            outputs = self.predictor(image)
            instances = outputs["instances"]
            scores = instances.scores.cpu().numpy()
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            pred_classes = instances.pred_classes.cpu().numpy()
            width, height = get_image_size(frame)

            out_det = []
            for idx in range(len(boxes)):
                xyxy = boxes[idx]
                box = [xyxy[0] / width, xyxy[1] / height, xyxy[2] / width, xyxy[3] / height]
                out_det.append(
                    {
                        "box": box,
                        "id": None,
                        "class": self.det_class_remap[pred_classes[idx]],
                        "confidence": scores[idx],
                    }
                )
            ret.append(out_det)
        return ret
