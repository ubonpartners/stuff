"""Ultralytics-based backend implementation."""

from __future__ import annotations

import concurrent.futures
import copy
from functools import partial

import cv2
import ultralytics

from ... import platform_stuff
from ... import ultralytics as ultralytics_stuff
from ...ultralytics import attributes_from_class_names
from ..base import InferenceBackend


class UltralyticsBackend(InferenceBackend):
    @staticmethod
    def _canonical_attr_name(name):
        """Normalize attribute naming variants like person:is_male vs person_is_male."""
        if not isinstance(name, str):
            return ""
        return name.strip().replace(":", "_")

    def __init__(self, model_name, model_params, config, class_synonyms):
        super().__init__()
        self.model_name = model_name
        self.model_params = model_params
        self.config = config
        self.class_synonyms = class_synonyms

        self.face_kp = config.face_kp
        self.pose_kp = config.pose_kp
        self.facepose_kp = config.facepose_kp
        self.fold_attributes = config.fold_attributes
        self.get_feats = config.get_feats

        # Ensure torchvision is loaded so Ultralytics uses fast NMS.
        import torchvision  # noqa: F401

        self.model = self._load_model(model_name)
        self.yolo_num_flops = self.yolo_num_flops * config.imgsz * config.imgsz / (640.0 * 640.0)

        self.num_threads = platform_stuff.platform_num_gpus()
        self.infer_batch_size = config.batch_size

        self.yolo = [None] * self.num_threads
        self.model_params = [dict(model_params) for _ in range(self.num_threads)]
        for i in range(self.num_threads):
            if i == 0:
                self.yolo[i] = self.model
            else:
                self.yolo[i] = self._load_model(model_name)
            if self.num_threads != 1:
                self.yolo[i] = self.yolo[i].to("cuda:" + str(i))
            if self.get_feats:
                self.yolo[i].add_callback("on_predict_start", partial(self._on_predict_start, persist=False))

        self.infer_model_class_names = [self.yolo[0].names[i] for i in range(len(self.yolo[0].names))]
        self.det_attributes_remap = None

        model_attr_names = getattr(getattr(self.yolo[0], "model", None), "attr_names", None)
        if self.fold_attributes and config.attributes is None:
            if model_attr_names is not None and len(model_attr_names) > 0:
                config.attributes = list(model_attr_names)
            else:
                config.attributes = attributes_from_class_names(self.infer_model_class_names)

        if model_attr_names is not None and config.attributes is not None:
            attr_index = {name: i for i, name in enumerate(config.attributes)}
            # Fallback map for schema variants where only delimiters differ (':' vs '_').
            canonical_attr_index = {
                self._canonical_attr_name(name): i for i, name in enumerate(config.attributes)
            }
            # -1 means model has attribute that target schema does not track.
            self.det_attributes_remap = []
            for name in model_attr_names:
                idx = attr_index.get(name, None)
                if idx is None:
                    idx = canonical_attr_index.get(self._canonical_attr_name(name), -1)
                self.det_attributes_remap.append(idx)

    def _on_predict_start(self, predictor: object, persist: bool = False) -> None:
        del persist
        is_end2end = bool(getattr(getattr(predictor, "model", None), "end2end", False))
        if is_end2end:
            predictor.save_feats = False
            predictor.expanded_feats = False
            predictor._feats = None
            predictor._feats2 = None
            return

        predictor.save_feats = True
        predictor.expanded_feats = True
        predictor._feats = None
        predictor._feats2 = None

        if getattr(predictor, "_feat_hooks_installed", False):
            return

        def pre_hook(module, input_data):
            del module
            predictor._feats = [tensor.detach() for tensor in input_data[0]]

        def post_hook(module, input_data, output_data):
            del module, input_data
            out0 = output_data[0] if isinstance(output_data, (tuple, list)) else output_data
            if hasattr(out0, "clone"):
                predictor._feats2 = out0.detach()

        head = predictor.model.model.model[-1]
        predictor._feat_pre_hook_handle = head.register_forward_pre_hook(pre_hook)
        predictor._feat_post_hook_handle = head.register_forward_hook(post_hook)
        predictor._feat_hooks_installed = True

    def _load_model(self, name):
        task = "detect"
        if "world" in name:
            base_classes = list(self.config.class_names or [])
            extended_classes = copy.deepcopy(base_classes)
            for class_name in base_classes:
                if class_name in self.class_synonyms:
                    for synonym in self.class_synonyms[class_name]:
                        if synonym not in extended_classes:
                            extended_classes.append(synonym)

            model = ultralytics.YOLOWorld(name, verbose=False)
            model.set_classes(extended_classes)
            return model

        if "yoloe" in name:
            model = ultralytics.YOLOE(name, task=task, verbose=False)
            class_names = self.config.class_names or []
            model.set_classes(class_names, model.get_text_pe(class_names))
            return model

        if "nas" in name:
            return ultralytics.NAS(name)

        if "rtdetr" in name:
            return ultralytics.RTDETR(name)

        if any(token in name for token in ("pose", "face", "full", "attributes", "dpa")):
            task = "pose"
        if "dpar" in name:
            task = "posereid"

        model = ultralytics.YOLO(name, task=task, verbose=False)
        info = None
        if not name.endswith(".engine"):
            info = model.info(verbose=False)
        self.yolo_num_params = 0
        self.yolo_num_flops = 0
        if info is not None:
            self.yolo_num_params = info[1]
            self.yolo_num_flops = info[3]
        return model

    def _flip_kp_inplace(self, keypoints):
        if keypoints is None:
            return
        for i in range(0, len(keypoints), 3):
            keypoints[i] = 1.0 - keypoints[i]

    def _process_batch(self, yolo_fn, frames, params):
        if params is not None and params.get("flip-lr", False):
            out_frames = []
            for frame in frames:
                if isinstance(frame, str):
                    frame = cv2.imread(frame)
                frame = cv2.flip(frame, 1)
                out_frames.append(frame)
            frames = out_frames

        kwargs = dict(
            conf=self.config.thr,
            iou=self.config.nms_thr,
            max_det=self.config.max_det,
            agnostic_nms=False,
            half=self.config.half,
            imgsz=self.config.imgsz,
            verbose=False,
            rect=self.config.rect,
        )
        if self.get_feats:
            kwargs["end2end"] = False
        return yolo_fn(frames, **kwargs)

    def infer(self, input_frames):
        results = []
        index = 0
        if self.num_threads == 1:
            while index < len(input_frames):
                end = min(len(input_frames), index + self.infer_batch_size)
                batch = input_frames[index:end]
                results += self._process_batch(self.yolo[0], batch, self.model_params[0])
                index = end
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                while index < len(input_frames):
                    futures = []
                    for i in range(self.num_threads):
                        end = min(len(input_frames), index + self.infer_batch_size)
                        if index < end:
                            batch = input_frames[index:end]
                            futures.append(executor.submit(self._process_batch, self.yolo[i], batch, self.model_params[i]))
                        index = end
                    for future in futures:
                        results += future.result()

        ret = []
        for result in results:
            out_det = ultralytics_stuff.yolo_results_to_dets(
                result,
                det_thr=self.config.thr,
                det_class_remap=self.det_class_remap,
                det_attributes_remap=self.det_attributes_remap,
                yolo_class_names=self.infer_model_class_names,
                class_names=self.class_names,
                attributes=self.attributes,
                face_kp=self.face_kp,
                pose_kp=self.pose_kp,
                facepose_kp=self.facepose_kp,
                fold_attributes=self.fold_attributes,
            )

            if self.model_params[0] is not None and self.model_params[0].get("flip-lr", False):
                for det in out_det:
                    x1, y1, x2, y2 = det["box"]
                    det["box"] = [1.0 - x2, y1, 1.0 - x1, y2]
                    if "pose_points" in det:
                        self._flip_kp_inplace(det["pose_points"])
                    if "face_points" in det:
                        self._flip_kp_inplace(det["face_points"])
                    if "facepose_points" in det:
                        self._flip_kp_inplace(det["facepose_points"])
            ret.append(out_det)
        return ret
