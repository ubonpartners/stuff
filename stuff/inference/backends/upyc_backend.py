"""UPYC/TRT-based backends."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
from PIL import Image

from ..base import InferenceBackend


class UPYCTrtBackend(InferenceBackend):
    def __init__(self, model_name, model_name_list, config):
        super().__init__()
        try:
            import ubon_pycstuff.ubon_pycstuff as upyc
        except ImportError as exc:
            raise AssertionError("TRT models require ubon_pycstuff") from exc

        self.upyc = upyc
        self.thr = config.thr
        self.fold_attributes = config.fold_attributes

        if not isinstance(model_name, str) or not model_name.endswith(".engine"):
            raise AssertionError(f"UPYC detector backend requires a .engine model, got: {model_name}")
        if len(model_name_list) > 1:
            raise AssertionError("UPYC .engine detector backend no longer accepts a second parameter")

        self.upyc_infer = upyc.c_infer(model_name)
        self.model_description = self.upyc_infer.get_model_description()
        self.model_description["engineInfo"] = json.loads(self.model_description["engineInfo"])

        self.infer_model_class_names = (
            self.model_description["class_names"] + self.model_description["person_attribute_names"]
        )
        self.first_attribute_class = len(self.model_description["class_names"])
        self.num_threads = 1
        self.infer_batch_size = config.batch_size

        self.upyc_infer.configure(
            {
                "det_thr": config.thr,
                "nms_thr": config.nms_thr,
                "max_detections": config.max_det,
                "allow_upscale": True,
            }
        )

    def infer(self, input_frames):
        from ubon_pycstuff.ubon_pycstuff import c_image

        assert len(input_frames) > 0, "Infer UPYC no input frames"
        images = []
        for frame in input_frames:
            if isinstance(frame, c_image):
                images.append(frame)
            elif isinstance(frame, str):
                images.append(self.upyc.load_jpeg(frame))
            elif isinstance(frame, Image.Image):
                np_img = np.array(frame.convert("RGB"))
                images.append(self.upyc.c_image.from_numpy(np_img))
            else:
                bgr_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(self.upyc.c_image.from_numpy(bgr_img))

        batch_dets = self.upyc_infer.run_batch(images)

        out_dets = []
        remap = self.det_class_remap
        conf_thresh = self.thr
        first_attr_class = self.first_attribute_class
        fold_attrs = self.fold_attributes

        for dets in batch_dets:
            processed_dets = []
            for det in dets:
                orig_class = remap[det["class"]]
                if orig_class != -1:
                    det["class"] = orig_class
                    processed_dets.append(det)

                if not fold_attrs and "attrs" in det:
                    base_det = {k: v for k, v in det.items() if k not in ("class", "confidence")}
                    for idx, conf in enumerate(det["attrs"]):
                        if conf > conf_thresh:
                            new_class = remap[first_attr_class + idx]
                            if new_class != -1:
                                det_copy = base_det.copy()
                                det_copy["class"] = new_class
                                det_copy["confidence"] = conf
                                processed_dets.append(det_copy)

            out_dets.append(processed_dets)
        return out_dets


class UPYCTrackBackend(InferenceBackend):
    def __init__(self, model_name_list, config):
        super().__init__()
        try:
            import ubon_pycstuff.ubon_pycstuff as upyc
        except ImportError as exc:
            raise AssertionError("upyc_track models require ubon_pycstuff") from exc

        self.upyc = upyc
        param_file = "/mldata/config/track/trackers/uc_reid.yaml"
        if len(model_name_list) > 1:
            param_file = model_name_list[1]

        self.track_shared = upyc.c_track_shared_state(param_file)
        self.track_stream = upyc.c_track_stream(self.track_shared)
        self.infer_model_class_names = ["person"]
        self.num_threads = 1
        self.infer_batch_size = config.batch_size

    def infer(self, input_frames):
        from ubon_pycstuff.ubon_pycstuff import c_image

        images = []
        for frame in input_frames:
            if isinstance(frame, c_image):
                images.append(frame)
            elif isinstance(frame, str):
                images.append(self.upyc.load_jpeg(frame))
            elif isinstance(frame, Image.Image):
                np_img = np.array(frame.convert("RGB"))
                images.append(self.upyc.c_image.from_numpy(np_img))
            else:
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(self.upyc.c_image.from_numpy(rgb_img))

        self.track_stream.run_on_individual_images(images)
        ret = []
        for i in range(500):
            track_results = self.track_stream.get_results(True)
            for result in track_results:
                ret.append(result["track_dets"])
            if len(ret) == len(input_frames):
                break
            if i > 100:
                time.sleep(0.005)
        assert len(ret) == len(input_frames), "Didn't get all results"
        return ret
