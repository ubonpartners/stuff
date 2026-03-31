"""Roboflow RF-DETR backend."""

from __future__ import annotations

from PIL import Image

from ...image import get_image_size
from ..base import InferenceBackend


class RFDetrBackend(InferenceBackend):
    def __init__(self, model_name, config):
        super().__init__()
        try:
            from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRSmall, RFDETRNano
            from rfdetr.util.coco_classes import COCO_CLASSES
        except ImportError as exc:
            raise AssertionError("try pip install rfdetr") from exc

        model_map = {
            "rfdetr-nano": (RFDETRNano, 16),
            "rfdetr-small": (RFDETRSmall, 16),
            "rfdetr-medium": (RFDETRMedium, 16),
            "rfdetr-large": (RFDETRLarge, 8),
            "rfdetr-base": (RFDETRBase, 16),
        }
        self.rf_detr_model = model_map[model_name][0](resolution=config.imgsz)
        self.rf_detr_batch_size = model_map[model_name][1]
        self.rf_detr_model.optimize_for_inference(batch_size=self.rf_detr_batch_size)

        num_classes = max([i for i in COCO_CLASSES]) + 1
        self.infer_model_class_names = ["-"] * num_classes
        for i in COCO_CLASSES:
            self.infer_model_class_names[i] = COCO_CLASSES[i]

        self.num_threads = 1
        self.infer_batch_size = self.rf_detr_batch_size
        self.thr = config.thr

    def infer(self, input_frames):
        detections_list = []
        for start in range(0, len(input_frames), self.rf_detr_batch_size):
            img_batch = list(input_frames[start : start + self.rf_detr_batch_size])
            for idx, img in enumerate(img_batch):
                if isinstance(img, str):
                    img_batch[idx] = Image.open(img).convert("RGB")
            real_batch_size = len(img_batch)
            while len(img_batch) < self.rf_detr_batch_size:
                img_batch.append(img_batch[-1])
            dets = self.rf_detr_model.predict(img_batch, threshold=self.thr)
            detections_list += dets[0:real_batch_size]

        ret = []
        for j, detections in enumerate(detections_list):
            out_det = []
            width, height = get_image_size(input_frames[j])
            for i in range(len(detections.xyxy)):
                xyxy = detections.xyxy[i]
                box = [xyxy[0] / width, xyxy[1] / height, xyxy[2] / width, xyxy[3] / height]
                cls = self.det_class_remap[int(detections.class_id[i] + 0.01)]
                det = {
                    "box": box,
                    "id": None,
                    "class": cls,
                    "confidence": detections.confidence[i],
                }
                if cls != -1:
                    out_det.append(det)
            ret.append(out_det)
        return ret
