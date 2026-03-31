"""Shared post-processing helpers for inference backends."""

from __future__ import annotations


CLASS_SYNONYMS = {
    "person": ["man", "woman", "boy", "girl"],
    "vehicle": ["car", "bicycle", "motorcycle", "train", "truck", "bus", "airplane", "boat"],
    "animal": ["cat", "dog", "horse", "sheep", "cow", "bird", "elephant", "bear", "zebra", "giraffe"],
    "weapon": ["gun", "dagger", "pistol", "handgun", "rifle", "revolver"],
    "face": [],
}


def merge_detections_wbf(detectors_outputs, iou_thr=0.55, skip_box_thr=0.001):
    from ensemble_boxes import weighted_boxes_fusion

    all_boxes = []
    all_scores = []
    all_labels = []

    for detections in detectors_outputs:
        boxes = []
        scores = []
        labels = []
        for det in detections:
            boxes.append(det["box"])
            scores.append(det["confidence"])
            labels.append(det["class"])
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        all_boxes,
        all_scores,
        all_labels,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    merged_detections = []
    for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
        merged_detections.append({"box": box, "confidence": score, "class": int(label)})

    return merged_detections


def ensemble_merge_results(results):
    num = len(results[0])
    out = []
    for i in range(num):
        dets = []
        for result in results:
            dets.append(result[i])
        out.append(merge_detections_wbf(dets))
    return out


def build_class_mapping(infer_model_class_names, class_names, class_synonyms=None):
    if class_synonyms is None:
        class_synonyms = CLASS_SYNONYMS

    if class_names is None:
        class_names = list(infer_model_class_names)

    det_class_remap = [-1] * len(infer_model_class_names)
    can_detect = [False] * len(class_names)

    for i, source_name in enumerate(infer_model_class_names):
        if source_name in class_names:
            dst_idx = class_names.index(source_name)
            det_class_remap[i] = dst_idx
            can_detect[dst_idx] = True
            continue
        for target_name, synonyms in class_synonyms.items():
            if target_name in class_names and source_name in synonyms:
                dst_idx = class_names.index(target_name)
                det_class_remap[i] = dst_idx
                can_detect[dst_idx] = True
                break

    return class_names, det_class_remap, can_detect
