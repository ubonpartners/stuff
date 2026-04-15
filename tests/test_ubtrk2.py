import os
import struct

import numpy as np
import yaml

from stuff.ubtrk2 import (
    UBTRK2Reader,
    UBTRK2Writer,
    decode_nested_payloads,
    encode_nested_arrays,
    is_ubtrk2_file,
)


def _box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), fourcc) + payload


def _pack_embedding_payload(values, emb_time=0.0, quality=0.0) -> bytes:
    payload = bytearray()
    payload.extend(struct.pack(">I", 1))  # embedding version
    payload.extend(struct.pack(">d", float(emb_time)))
    payload.extend(struct.pack(">f", float(quality)))
    payload.extend(struct.pack(">I", len(values)))
    for v in values:
        payload.extend(struct.pack(">f", float(v)))
    return bytes(payload)


def _pack_c_detection_list_payload(version: int, include_embeddings: bool) -> bytes:
    assert version in (1, 2)
    payload = bytearray()
    payload.extend(struct.pack(">I", version))
    payload.extend(struct.pack(">d", 0.04))
    payload.extend(struct.pack(">I", 1))  # num detections

    if version >= 2:
        if include_embeddings:
            frame_clip = _pack_embedding_payload([0.9, 0.8, 0.7], emb_time=0.04, quality=0.99)
            payload.extend(struct.pack(">B", 1))
            payload.extend(struct.pack(">I", len(frame_clip)))
            payload.extend(frame_clip)
        else:
            payload.extend(struct.pack(">B", 0))

    # Core detection fields
    payload.extend(struct.pack(">5f", 0.1, 0.2, 0.4, 0.6, 0.95))  # box + conf
    payload.extend(struct.pack(">d", 0.04))  # last_seen_time
    payload.extend(struct.pack(">H", 0))  # class
    payload.extend(struct.pack(">H", 3))  # index
    payload.extend(struct.pack(">Q", 42))  # track_id
    payload.extend(struct.pack(">Q", 1234))  # overlap mask
    payload.extend(struct.pack(">4f", 0.12, 0.22, 0.24, 0.34))  # subbox
    payload.extend(struct.pack(">f", 0.85))  # subbox conf
    payload.extend(struct.pack(">f", 0.77))  # fiqa
    payload.extend(struct.pack(">B", 2))  # num_face_points
    payload.extend(struct.pack(">B", 3))  # num_pose_points
    payload.extend(struct.pack(">B", 2))  # num_attr
    payload.extend(struct.pack(">B", 4))  # reid_vector_len

    face_points = np.zeros((5, 3), dtype=np.float32)
    face_points[0] = [0.15, 0.25, 0.9]
    face_points[1] = [0.20, 0.27, 0.8]
    payload.extend(face_points.astype(">f4").tobytes())

    pose_points = np.zeros((17, 3), dtype=np.float32)
    pose_points[0] = [0.2, 0.3, 0.7]
    pose_points[1] = [0.25, 0.35, 0.6]
    pose_points[2] = [0.3, 0.4, 0.5]
    payload.extend(pose_points.astype(">f4").tobytes())

    attrs = np.zeros((64,), dtype=np.float32)
    attrs[0] = 0.1
    attrs[1] = 0.2
    payload.extend(attrs.astype(">f4").tobytes())

    reid = np.zeros((80,), dtype=np.float32)
    reid[:4] = [1.0, 2.0, 3.0, 4.0]
    payload.extend(reid.astype(">f4").tobytes())

    if version >= 2:
        if include_embeddings:
            face_emb = _pack_embedding_payload([0.5, 0.4, 0.3], emb_time=0.04, quality=0.91)
            payload.extend(struct.pack(">B", 1))
            payload.extend(struct.pack(">I", len(face_emb)))
            payload.extend(face_emb)
            clip_emb = _pack_embedding_payload([0.2, 0.1, 0.0], emb_time=0.04, quality=0.81)
            payload.extend(struct.pack(">B", 1))
            payload.extend(struct.pack(">I", len(clip_emb)))
            payload.extend(clip_emb)
        else:
            payload.extend(struct.pack(">B", 0))
            payload.extend(struct.pack(">B", 0))

    return bytes(payload)


def _write_c_style_ubtrk2_file(path: str, dets_version: int, include_embeddings: bool) -> None:
    metadata = {
        "schema_version": 2,
        "kind": "trackset",
        "container": "UBTRK2",
        "format": "uboncstuff-track-v1",
        "frame_rate": 25.0,
        "width": 1920,
        "height": 1080,
        "classes": ["person", "face"],
    }
    fhdr_payload = struct.pack(
        ">IdIf8f",
        1,  # frame header version
        0.04,
        3,  # tracked_roi
        0.12,
        0.1, 0.2, 0.9, 0.95,  # motion_roi
        0.0, 0.0, 1.0, 1.0,  # inference_roi
    )
    trks_payload = _pack_c_detection_list_payload(dets_version, include_embeddings=include_embeddings)
    dets_payload = _pack_c_detection_list_payload(dets_version, include_embeddings=include_embeddings)

    frame_payload = (
        _box(b"fhdr", fhdr_payload)
        + _box(b"trks", trks_payload)
        + _box(b"dets", dets_payload)
    )
    with open(path, "wb") as fp:
        fp.write(_box(b"ubtf", struct.pack(">HH", 2, 0)))
        fp.write(_box(b"meta", yaml.safe_dump(metadata, sort_keys=True).encode("utf-8")))
        fp.write(_box(b"fram", frame_payload))


def test_nested_array_payload_roundtrip():
    payload = {
        "a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "b": [np.array([5, 6, 7], dtype=np.int16)],
    }
    encoded = encode_nested_arrays(payload)
    decoded = decode_nested_payloads(encoded)
    assert isinstance(decoded["a"], np.ndarray)
    assert decoded["a"].dtype == np.float32
    assert decoded["a"].shape == (2, 2)
    np.testing.assert_allclose(decoded["a"], payload["a"])
    assert isinstance(decoded["b"][0], np.ndarray)
    np.testing.assert_array_equal(decoded["b"][0], payload["b"][0])


def test_ubtrk2_file_roundtrip(tmp_path):
    output = tmp_path / "run.ubtrk2"
    metadata = {
        "schema_version": 2,
        "kind": "trackset",
        "frame_rate": 25.0,
        "width": 1920,
        "height": 1080,
        "classes": ["person", "face"],
    }
    frame0 = {
        "frame_time": 0.04,
        "result_type": "tracked_roi",
        "motion_score": 0.12,
        "motion_roi": [0.1, 0.2, 0.9, 0.95],
        "inference_roi": [0.0, 0.0, 1.0, 1.0],
        "objects": {
            1001: {
                "class": 0,
                "confidence": 0.95,
                "box": [0.1, 0.1, 0.2, 0.3],
                "reid_vector": {
                    "__payload_kind__": "ndarray",
                    "dtype": "float32",
                    "shape": [4],
                    "codec": "raw",
                    "data": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes(),
                },
            }
        },
        "debug": {
            "motion_field": {
                "type": "motion_field",
                "data": {
                    "flow": {
                        "__payload_kind__": "ndarray",
                        "dtype": "float32",
                        "shape": [2, 2, 2],
                        "codec": "raw",
                        "data": np.zeros((2, 2, 2), dtype=np.float32).tobytes(),
                    }
                },
            }
        },
        "image_path": "/tmp/frame_0001.jpg",
    }
    frame1 = {
        "frame_time": 0.08,
        "result_type": "skip_no_motion",
        "motion_score": 0.0,
        "motion_roi": [0.0, 0.0, 0.0, 0.0],
        "inference_roi": [0.0, 0.0, 0.0, 0.0],
        "objects": None,
        "debug": None,
    }

    with UBTRK2Writer(str(output), metadata) as writer:
        writer.write_frame(frame0)
        writer.write_frame(frame1)

    assert os.path.isfile(output)
    assert is_ubtrk2_file(str(output))

    reader = UBTRK2Reader(str(output))
    assert reader.num_frames == 2
    assert reader.metadata["kind"] == "trackset"
    frames = list(reader.iter_frames())
    assert frames[0]["result_type"] == "tracked_roi"
    assert frames[1]["result_type"] == "skip_no_motion"
    reid_obj = frames[0]["objects"].get(1001) or frames[0]["objects"].get("1001")
    assert reid_obj is not None
    reid = reid_obj["reid_vector"]
    assert isinstance(reid, np.ndarray)
    np.testing.assert_allclose(reid, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    flow = frames[0]["debug"]["motion_field"]["data"]["flow"]
    assert isinstance(flow, np.ndarray)
    assert flow.shape == (2, 2, 2)


def test_c_style_dets_v1_decode_compat(tmp_path):
    output = tmp_path / "run_c_v1.ubtrk2"
    _write_c_style_ubtrk2_file(str(output), dets_version=1, include_embeddings=False)

    reader = UBTRK2Reader(str(output))
    frames = list(reader.iter_frames())
    assert len(frames) == 1
    f0 = frames[0]
    assert f0["result_type"] == "tracked_roi"
    assert isinstance(f0["objects"], dict)
    obj = f0["objects"].get(42) or f0["objects"].get("42")
    assert obj is not None
    assert "clip_embedding" not in obj
    assert "face_embedding" not in obj
    assert f0.get("clip_embedding") is None
    assert isinstance(f0["inference_dets"], list)
    assert len(f0["inference_dets"]) == 1


def test_c_style_dets_v2_clip_embedding_decode(tmp_path):
    output = tmp_path / "run_c_v2.ubtrk2"
    _write_c_style_ubtrk2_file(str(output), dets_version=2, include_embeddings=True)

    reader = UBTRK2Reader(str(output))
    frames = list(reader.iter_frames())
    assert len(frames) == 1
    f0 = frames[0]
    assert f0["result_type"] == "tracked_roi"
    frame_clip = f0.get("clip_embedding")
    assert isinstance(frame_clip, np.ndarray)
    np.testing.assert_allclose(frame_clip, np.array([0.9, 0.8, 0.7], dtype=np.float32))

    obj = f0["objects"].get(42) or f0["objects"].get("42")
    assert obj is not None
    assert isinstance(obj.get("face_embedding"), np.ndarray)
    assert isinstance(obj.get("clip_embedding"), np.ndarray)
    np.testing.assert_allclose(obj["face_embedding"], np.array([0.5, 0.4, 0.3], dtype=np.float32))
    np.testing.assert_allclose(obj["clip_embedding"], np.array([0.2, 0.1, 0.0], dtype=np.float32))

    det = f0["inference_dets"][0]
    assert isinstance(det.get("face_embedding"), np.ndarray)
    assert isinstance(det.get("clip_embedding"), np.ndarray)

