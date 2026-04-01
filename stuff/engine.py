"""Shared helpers for metadata-wrapped TensorRT .engine files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ENGINE_ARG_KEYS = ("batch", "dynamic", "half", "int8", "simplify", "nms", "fraction")


def write_engine_with_metadata(dst_engine: str | Path, trt_engine_bytes: bytes, metadata: dict) -> None:
    """Write Ultralytics-compatible engine header + TRT payload."""
    dst_engine = Path(dst_engine)
    meta_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    meta_raw = meta_json.encode("utf-8")
    with dst_engine.open("wb") as f:
        f.write(len(meta_raw).to_bytes(4, byteorder="little", signed=False))
        f.write(meta_raw)
        f.write(trt_engine_bytes)


def read_engine_metadata(engine_path: str | Path) -> tuple[dict, int, int]:
    """Read embedded metadata from a metadata-wrapped .engine file."""
    engine_path = Path(engine_path)
    with engine_path.open("rb") as f:
        raw_len = f.read(4)
        if len(raw_len) != 4:
            raise ValueError("File too small to contain engine metadata")
        meta_len = int.from_bytes(raw_len, byteorder="little", signed=False)
        if meta_len <= 0:
            raise ValueError(f"Invalid metadata length: {meta_len}")
        raw_meta = f.read(meta_len)
        if len(raw_meta) != meta_len:
            raise ValueError("Truncated metadata payload")
    metadata = json.loads(raw_meta.decode("utf-8"))
    payload_size = engine_path.stat().st_size - 4 - meta_len
    return metadata, meta_len, payload_size


def _infer_face_person_kpts(num_kpts: int) -> tuple[int, int]:
    if num_kpts == 22:
        return 5, 17
    if num_kpts == 17:
        return 0, 17
    if num_kpts == 19:
        return 0, 19
    if num_kpts == 5:
        return 5, 0
    return 0, max(0, num_kpts)


def _infer_reid_vector_len(head) -> int:
    reid = getattr(head, "reid", None)
    if reid is None:
        return 0
    for key in ("emb", "emb_dim", "out_dim"):
        v = getattr(reid, key, None)
        if isinstance(v, int) and v > 0:
            return v
    return 0


def _normalize_class_names(raw_names) -> dict[int, str]:
    if isinstance(raw_names, dict):
        pairs = []
        for k, v in raw_names.items():
            try:
                idx = int(k)
            except Exception:
                continue
            pairs.append((idx, str(v)))
        pairs.sort(key=lambda x: x[0])
        return {idx: name for idx, name in pairs}
    if isinstance(raw_names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(raw_names)}
    return {}


def _normalize_name_list(raw_names) -> list[str]:
    if isinstance(raw_names, (list, tuple)):
        return [str(x) for x in raw_names]
    if isinstance(raw_names, dict):
        pairs = []
        for k, v in raw_names.items():
            try:
                idx = int(k)
            except Exception:
                continue
            pairs.append((idx, str(v)))
        pairs.sort(key=lambda x: x[0])
        if pairs and [p[0] for p in pairs] == list(range(len(pairs))):
            return [p[1] for p in pairs]
    return []


def _normalize_kpt_names(raw_kpt_names, num_kpts: int) -> list[str] | None:
    if num_kpts <= 0 or raw_kpt_names is None:
        return None
    # Common simple form: ["nose", "left_eye", ...]
    if isinstance(raw_kpt_names, (list, tuple)):
        names = [str(x) for x in raw_kpt_names]
        return names if len(names) == num_kpts else None
    # Ultralytics can emit per-class mapping or indexed map.
    if isinstance(raw_kpt_names, dict):
        # Case 1: {0:"a",1:"b",...}
        idx_names = _normalize_name_list(raw_kpt_names)
        if len(idx_names) == num_kpts:
            return idx_names
        # Case 2: {0:[...], 1:[...], ...}; keep one if all classes share same list.
        seq_values = []
        for v in raw_kpt_names.values():
            if isinstance(v, (list, tuple)):
                seq_values.append([str(x) for x in v])
        if seq_values:
            first = seq_values[0]
            if len(first) == num_kpts and all(v == first for v in seq_values):
                return first
    return None


def _load_dataset_names_from_yaml(data_path: str) -> tuple[dict[int, str], list[str]]:
    if not data_path:
        return {}, []
    p = Path(data_path)
    if not p.exists():
        return {}, []
    try:
        import yaml
    except Exception:
        return {}, []
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}, []
    return (
        _normalize_class_names(data.get("names")),
        _normalize_name_list(data.get("attr_names")),
    )


def build_yolo_engine_metadata_from_pt(
    pt_path: str | Path,
    *,
    engine_args: dict | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Build rich engine metadata from a YOLO .pt checkpoint."""
    import torch
    import ultralytics

    from ultralytics import __version__ as ultralytics_version

    pt_path = Path(pt_path)
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    ta = ckpt.get("train_args", {})
    if hasattr(ta, "__dict__"):
        ta = vars(ta)

    yolo = ultralytics.YOLO(str(pt_path))
    model = yolo.model
    head = model.model[-1] if hasattr(model, "model") and len(model.model) else None

    names = _normalize_class_names(yolo.names)
    stride = int(max(model.stride)) if hasattr(model, "stride") else 32
    task = getattr(model, "task", "detect")
    channels = int(getattr(model, "yaml", {}).get("channels", 3))
    data_name = ta.get("data", "")
    pretty_name = pt_path.stem.replace("yolo", "YOLO")
    description = f"Ultralytics {pretty_name} model {f'trained on {data_name}' if data_name else ''}".strip()

    yaml_names, yaml_attr_names = _load_dataset_names_from_yaml(str(data_name))
    if not names and yaml_names:
        names = yaml_names

    nc = int(ta.get("nc", len(names)))
    nc_attr = int(ta.get("nc_attr", getattr(head, "attr_nc", 0) or 0))
    attr_names = _normalize_name_list(ta.get("attr_names"))
    if not attr_names:
        attr_names = _normalize_name_list(getattr(model, "attr_names", None))
    if not attr_names:
        attr_names = _normalize_name_list(getattr(head, "attr_names", None))
    if not attr_names and yaml_attr_names:
        attr_names = yaml_attr_names
    # If nc_attr is known but names are unavailable, keep metadata internally consistent.
    if nc_attr > 0 and not attr_names:
        attr_names = [f"attr_{i}" for i in range(nc_attr)]
    if nc_attr <= 0 and attr_names:
        nc_attr = len(attr_names)
    if nc_attr > 0 and len(attr_names) > nc_attr:
        attr_names = attr_names[:nc_attr]

    kpt_shape = list(ta.get("kpt_shape", list(getattr(head, "kpt_shape", [0, 3]))))
    end2end = bool(ta.get("end2end", getattr(model, "end2end", False)))
    num_kpts = int(kpt_shape[0]) if kpt_shape else 0
    face_kpts, person_kpts = _infer_face_person_kpts(num_kpts)
    reid_vector_len = _infer_reid_vector_len(head)

    pt_size_mb = round(pt_path.stat().st_size / (1024 * 1024), 1)
    ema = ckpt.get("ema")
    if ema is not None and hasattr(ema, "parameters"):
        n_params = sum(p.numel() for p in ema.parameters())
    else:
        n_params = sum(p.numel() for p in model.parameters()) if model is not None else 0

    training = {}
    if ckpt.get("date"):
        training["date"] = ckpt["date"]
    if ckpt.get("version"):
        training["ultralytics_version"] = ckpt["version"]
    git = ckpt.get("git", {})
    if git.get("branch"):
        training["git_branch"] = git["branch"]
    if git.get("commit"):
        training["git_commit"] = git["commit"][:7]
    final_epoch = ckpt.get("epoch")
    if final_epoch is not None:
        training["epoch"] = int(final_epoch) + 1
    if ta.get("epochs"):
        training["epochs_configured"] = int(ta["epochs"])
    if ckpt.get("best_fitness") is not None:
        training["best_fitness"] = round(float(ckpt["best_fitness"]), 4)
    if ta.get("imgsz") is not None:
        training["imgsz"] = ta["imgsz"]
    if ta.get("batch") is not None and int(ta["batch"]) > 0:
        training["batch"] = int(ta["batch"])
    if ta.get("optimizer"):
        training["optimizer"] = ta["optimizer"]
    if ta.get("lr0") is not None:
        training["lr0"] = float(ta["lr0"])
    raw_metrics = ckpt.get("train_metrics", {})
    metrics = {k: round(float(v), 4) for k, v in raw_metrics.items() if isinstance(v, (int, float)) and k != "fitness"}
    if metrics:
        training["metrics"] = metrics

    if engine_args is None:
        engine_args = {}
    args_subset = {k: engine_args[k] for k in ENGINE_ARG_KEYS if k in engine_args}

    metadata = {
        "description": description,
        "author": "Ultralytics",
        "date": datetime.now().isoformat(),
        "version": ultralytics_version,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "stride": stride,
        "task": task,
        "batch": int(engine_args.get("batch", 1)),
        "imgsz": engine_args.get("imgsz", ta.get("imgsz", [640, 640])),
        "names": names,
        "args": args_subset,
        "channels": channels,
        "end2end": end2end,
        # Custom extras needed in Ubon pipelines.
        "arch": pt_path.stem,
        "nc": nc,
        "nc_attr": nc_attr,
        "attr_names": attr_names,
        "face_kpts": face_kpts,
        "person_kpts": person_kpts,
        "reid_vector_len": reid_vector_len,
        "model_info": {
            "parameters": int(n_params),
            "parameters_M": round(n_params / 1e6, 1),
            "pt_size_mb": pt_size_mb,
        },
        "training": training,
    }
    if num_kpts > 0:
        metadata["kpt_shape"] = kpt_shape
        kpt_names = _normalize_kpt_names(getattr(model, "kpt_names", None), num_kpts)
        if kpt_names is None:
            kpt_names = _normalize_kpt_names(getattr(head, "kpt_names", None), num_kpts)
        if kpt_names is not None:
            metadata["kpt_names"] = kpt_names
    if "dla" in engine_args and engine_args["dla"] is not None:
        metadata["dla"] = str(engine_args["dla"])
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata
