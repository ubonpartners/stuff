import os

import numpy as np

from stuff.ubtrk2 import (
    UBTRK2Reader,
    UBTRK2Writer,
    decode_nested_payloads,
    encode_nested_arrays,
    is_ubtrk2_file,
)


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

