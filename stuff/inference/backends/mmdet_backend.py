"""MMDetection backend."""

from __future__ import annotations

import cv2

from ...image import get_image_size
from ..base import InferenceBackend


class MMDetBackend(InferenceBackend):
    def __init__(self, model_name, config):
        super().__init__()
        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError as exc:
            raise AssertionError("try pip install 'mmdet'") from exc

        mmdet_path = "/home/mark/stuff/ai/mmdetection/configs"
        mmdet_models = {
            "cascade-mask-rcnn_x101-64x4d_fpn_ms-3x": (
                mmdet_path + "/cascade_rcnn/cascade-mask-rcnn_x101-64x4d_fpn_ms-3x_coco.py",
                "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth",
            ),
            "grounding-dino-b": (
                mmdet_path + "/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py",
                "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201-f219e0c0.pth",
            ),
            "dino-5scale": (
                mmdet_path + "/dino/dino-5scale_swin-l_8xb2-36e_coco.py",
                "https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth",
            ),
            "crowddet2": (
                mmdet_path + "/crowddet/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman.py",
                "https://download.openmmlab.com/mmdetection/v3.0/crowddet/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman_20221023_174954-dc319c2d.pth",
            ),
        }

        cfg_path, weight_url = mmdet_models[model_name]
        self.inference_detector = inference_detector
        self.model = init_detector(cfg_path, weight_url, device="cuda:0")
        self.infer_model_class_names = self.model.dataset_meta["classes"]
        self.thr = config.thr
        self.num_threads = 1
        self.infer_batch_size = config.batch_size

    def infer(self, input_frames):
        ret = []
        for img in input_frames:
            if isinstance(img, str):
                image = cv2.imread(img)
            else:
                image = img

            width, height = get_image_size(image)
            det_sample = self.inference_detector(self.model, image)
            instances = det_sample.pred_instances
            boxes = instances.bboxes.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            labels = instances.labels.cpu().numpy()

            out_det = []
            for box, score, label in zip(boxes, scores, labels):
                if score < self.thr:
                    continue
                box_norm = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
                out_det.append(
                    {
                        "box": box_norm,
                        "id": None,
                        "class": self.det_class_remap[int(label)],
                        "confidence": score,
                    }
                )
            ret.append(out_det)
        return ret
