"""UBTRK2 tracker-run container utilities.

UBTRK2 is a BMFF-like 4CC + length-prefixed container designed for
efficient append/read in both Python and C/C++.

Container layout:
    [ubtf box]  file/version header
    [meta box]  UTF-8 YAML metadata payload
    [fram box]* repeated frame boxes, each containing child boxes

Each box uses:
    uint32_be box_size_including_header
    char[4]   fourcc
    bytes     payload
"""

from __future__ import annotations

import io
import math
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is optional for read-only metadata use
    np = None


UBTRK2_MAJOR_VERSION = 2
UBTRK2_MINOR_VERSION = 0

FOURCC_HEADER = b"ubtf"
FOURCC_META = b"meta"
FOURCC_FRAME = b"fram"

FOURCC_FRAME_HEADER = b"fhdr"
FOURCC_TRACKS = b"trks"
FOURCC_DETECTIONS = b"dets"
FOURCC_DEBUG = b"dbug"
FOURCC_IMAGE_PATH = b"imgp"
FOURCC_EXTRA = b"xtra"

CODEC_RAW = "raw"

_TYPE_NONE = 0x00
_TYPE_FALSE = 0x01
_TYPE_TRUE = 0x02
_TYPE_INT64 = 0x03
_TYPE_FLOAT64 = 0x04
_TYPE_STR = 0x05
_TYPE_BYTES = 0x06
_TYPE_LIST = 0x07
_TYPE_DICT = 0x08


class UBTRK2Error(RuntimeError):
    """Raised for malformed UBTRK2 content."""


@dataclass
class BoxHeader:
    fourcc: bytes
    payload_size: int
    total_size: int


def _is_fourcc(data: bytes) -> bool:
    return isinstance(data, (bytes, bytearray)) and len(data) == 4


def _read_exact(fp: io.BufferedReader, size: int) -> bytes:
    data = fp.read(size)
    if data is None or len(data) != size:
        raise UBTRK2Error(f"Unexpected EOF while reading {size} bytes")
    return data


def _read_box_header(fp: io.BufferedReader) -> Optional[BoxHeader]:
    raw = fp.read(8)
    if raw == b"":
        return None
    if len(raw) != 8:
        raise UBTRK2Error("Truncated box header")
    total_size, fourcc = struct.unpack(">I4s", raw)
    if total_size < 8:
        raise UBTRK2Error(f"Invalid box size {total_size} for {fourcc!r}")
    return BoxHeader(fourcc=fourcc, payload_size=total_size - 8, total_size=total_size)


def _write_box(fp: io.BufferedWriter, fourcc: bytes, payload: bytes) -> None:
    if not _is_fourcc(fourcc):
        raise ValueError(f"Invalid fourcc {fourcc!r}")
    total_size = 8 + len(payload)
    if total_size > 0xFFFFFFFF:
        raise ValueError(f"Box {fourcc!r} too large: {total_size} bytes")
    fp.write(struct.pack(">I4s", total_size, fourcc))
    fp.write(payload)


def _iter_boxes(payload: bytes) -> Iterator[Tuple[bytes, bytes]]:
    offset = 0
    size = len(payload)
    while offset < size:
        if offset + 8 > size:
            raise UBTRK2Error("Truncated child box header")
        total_size, fourcc = struct.unpack(">I4s", payload[offset : offset + 8])
        if total_size < 8:
            raise UBTRK2Error(f"Invalid child box size {total_size} for {fourcc!r}")
        end = offset + total_size
        if end > size:
            raise UBTRK2Error(f"Child box {fourcc!r} exceeds parent bounds")
        yield fourcc, payload[offset + 8 : end]
        offset = end


def _pack_u32(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"Length out of range: {value}")
    return struct.pack(">I", value)


def _unpack_u32(data: bytes, offset: int) -> Tuple[int, int]:
    if offset + 4 > len(data):
        raise UBTRK2Error("Unexpected EOF while reading uint32")
    return struct.unpack(">I", data[offset : offset + 4])[0], offset + 4


def _encode_value(value: Any) -> bytes:
    if value is None:
        return bytes([_TYPE_NONE])
    if value is False:
        return bytes([_TYPE_FALSE])
    if value is True:
        return bytes([_TYPE_TRUE])
    if isinstance(value, int) and not isinstance(value, bool):
        return bytes([_TYPE_INT64]) + struct.pack(">q", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float values are not supported")
        return bytes([_TYPE_FLOAT64]) + struct.pack(">d", value)
    if isinstance(value, str):
        data = value.encode("utf-8")
        return bytes([_TYPE_STR]) + _pack_u32(len(data)) + data
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return bytes([_TYPE_BYTES]) + _pack_u32(len(data)) + data
    if isinstance(value, (list, tuple)):
        out = bytearray()
        out.append(_TYPE_LIST)
        out.extend(_pack_u32(len(value)))
        for item in value:
            encoded = _encode_value(item)
            out.extend(_pack_u32(len(encoded)))
            out.extend(encoded)
        return bytes(out)
    if isinstance(value, dict):
        out = bytearray()
        out.append(_TYPE_DICT)
        out.extend(_pack_u32(len(value)))
        for key, item in value.items():
            if isinstance(key, int) and not isinstance(key, bool):
                key = str(key)
            if not isinstance(key, str):
                raise ValueError(f"Only string/int dict keys are supported, got {type(key)}")
            key_b = key.encode("utf-8")
            encoded = _encode_value(item)
            out.extend(_pack_u32(len(key_b)))
            out.extend(key_b)
            out.extend(_pack_u32(len(encoded)))
            out.extend(encoded)
        return bytes(out)
    raise TypeError(f"Unsupported value type for UBTRK2 codec: {type(value)}")


def _decode_value(data: bytes, offset: int = 0) -> Tuple[Any, int]:
    if offset >= len(data):
        raise UBTRK2Error("Unexpected EOF while decoding value")
    t = data[offset]
    offset += 1
    if t == _TYPE_NONE:
        return None, offset
    if t == _TYPE_FALSE:
        return False, offset
    if t == _TYPE_TRUE:
        return True, offset
    if t == _TYPE_INT64:
        if offset + 8 > len(data):
            raise UBTRK2Error("Truncated int64")
        return struct.unpack(">q", data[offset : offset + 8])[0], offset + 8
    if t == _TYPE_FLOAT64:
        if offset + 8 > len(data):
            raise UBTRK2Error("Truncated float64")
        return struct.unpack(">d", data[offset : offset + 8])[0], offset + 8
    if t in (_TYPE_STR, _TYPE_BYTES):
        length, offset = _unpack_u32(data, offset)
        end = offset + length
        if end > len(data):
            raise UBTRK2Error("Truncated bytes/string")
        raw = data[offset:end]
        if t == _TYPE_STR:
            return raw.decode("utf-8"), end
        return raw, end
    if t == _TYPE_LIST:
        count, offset = _unpack_u32(data, offset)
        out: List[Any] = []
        for _ in range(count):
            elem_len, offset = _unpack_u32(data, offset)
            end = offset + elem_len
            if end > len(data):
                raise UBTRK2Error("Truncated list element")
            item, consumed = _decode_value(data[offset:end], 0)
            if consumed != elem_len:
                raise UBTRK2Error("List element did not decode fully")
            out.append(item)
            offset = end
        return out, offset
    if t == _TYPE_DICT:
        count, offset = _unpack_u32(data, offset)
        out: Dict[str, Any] = {}
        for _ in range(count):
            key_len, offset = _unpack_u32(data, offset)
            key_end = offset + key_len
            if key_end > len(data):
                raise UBTRK2Error("Truncated dict key")
            key = data[offset:key_end].decode("utf-8")
            offset = key_end
            val_len, offset = _unpack_u32(data, offset)
            val_end = offset + val_len
            if val_end > len(data):
                raise UBTRK2Error("Truncated dict value")
            item, consumed = _decode_value(data[offset:val_end], 0)
            if consumed != val_len:
                raise UBTRK2Error("Dict value did not decode fully")
            out[key] = item
            offset = val_end
        return out, offset
    raise UBTRK2Error(f"Unknown value type tag 0x{t:02x}")


def encode_nested_arrays(value: Any) -> Any:
    """Recursively convert numpy arrays into serializable payload wrappers."""

    if np is not None and isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "__payload_kind__": "ndarray",
            "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape],
            "codec": CODEC_RAW,
            "data": arr.tobytes(order="C"),
        }
    if isinstance(value, dict):
        return {k: encode_nested_arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode_nested_arrays(v) for v in value]
    if isinstance(value, tuple):
        return [encode_nested_arrays(v) for v in value]
    return value


def decode_payload(payload: Any) -> Any:
    """Decode one payload wrapper object into bytes/numpy."""

    if not isinstance(payload, dict):
        return payload
    kind = payload.get("__payload_kind__")
    if kind == "bytes":
        return payload.get("data")
    if kind != "ndarray":
        return payload
    if np is None:
        return payload
    dtype = payload.get("dtype")
    shape = payload.get("shape")
    data = payload.get("data")
    codec = payload.get("codec", CODEC_RAW)
    if codec != CODEC_RAW:
        raise UBTRK2Error(f"Unsupported ndarray codec {codec!r}")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise UBTRK2Error("ndarray payload has non-bytes data")
    arr = np.frombuffer(bytes(data), dtype=np.dtype(dtype))
    return arr.reshape(tuple(int(x) for x in shape))


def decode_nested_payloads(value: Any) -> Any:
    """Recursively decode payload wrappers in dict/list trees."""

    decoded = decode_payload(value)
    if decoded is not value:
        return decoded
    if isinstance(value, dict):
        return {k: decode_nested_payloads(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_nested_payloads(v) for v in value]
    return value


def is_ubtrk2_file(path: str) -> bool:
    """Return True if file begins with a valid UBTRK2 header box."""

    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as fp:
            header = _read_box_header(fp)
            if header is None or header.fourcc != FOURCC_HEADER:
                return False
            payload = _read_exact(fp, header.payload_size)
            major, _minor = struct.unpack(">HH", payload)
            return major == UBTRK2_MAJOR_VERSION
    except Exception:
        return False


class UBTRK2Writer:
    """Streaming UBTRK2 writer."""

    def __init__(self, path: str, metadata: Dict[str, Any]):
        self.path = path
        self.metadata = metadata
        self._fp: Optional[io.BufferedWriter] = None

    def __enter__(self) -> "UBTRK2Writer":
        self._fp = open(self.path, "wb")
        header_payload = struct.pack(">HH", UBTRK2_MAJOR_VERSION, UBTRK2_MINOR_VERSION)
        _write_box(self._fp, FOURCC_HEADER, header_payload)
        meta_payload = yaml.safe_dump(self.metadata, sort_keys=True).encode("utf-8")
        _write_box(self._fp, FOURCC_META, meta_payload)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def write_frame(self, frame: Dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("UBTRK2Writer is not open")
        frame_payload = _encode_frame_box_payload(frame)
        _write_box(self._fp, FOURCC_FRAME, frame_payload)


class UBTRK2Reader:
    """Indexed UBTRK2 reader with frame iteration/random access."""

    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, Any] = {}
        self.version: Tuple[int, int] = (0, 0)
        self._frame_offsets: List[Tuple[int, int]] = []
        self._index_file()

    def _index_file(self) -> None:
        with open(self.path, "rb") as fp:
            while True:
                box_start = fp.tell()
                header = _read_box_header(fp)
                if header is None:
                    break
                if header.fourcc == FOURCC_HEADER:
                    payload = _read_exact(fp, header.payload_size)
                    if len(payload) != 4:
                        raise UBTRK2Error("Invalid ubtf payload length")
                    self.version = struct.unpack(">HH", payload)
                    if self.version[0] != UBTRK2_MAJOR_VERSION:
                        raise UBTRK2Error(
                            f"Unsupported UBTRK2 major version {self.version[0]}"
                        )
                elif header.fourcc == FOURCC_META:
                    payload = _read_exact(fp, header.payload_size)
                    loaded = yaml.safe_load(payload.decode("utf-8"))
                    self.metadata = loaded if isinstance(loaded, dict) else {}
                elif header.fourcc == FOURCC_FRAME:
                    self._frame_offsets.append((fp.tell(), header.payload_size))
                    fp.seek(header.payload_size, io.SEEK_CUR)
                else:
                    # Unknown top-level box; skip for forward compatibility.
                    fp.seek(header.payload_size, io.SEEK_CUR)
                if fp.tell() <= box_start:
                    raise UBTRK2Error("Reader made no progress while indexing")

    @property
    def num_frames(self) -> int:
        return len(self._frame_offsets)

    def iter_frames(self) -> Iterator[Dict[str, Any]]:
        with open(self.path, "rb") as fp:
            for offset, payload_size in self._frame_offsets:
                fp.seek(offset)
                payload = _read_exact(fp, payload_size)
                yield _decode_frame_box_payload(payload)

    def read_frame(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self._frame_offsets):
            raise IndexError(f"Frame index out of range: {index}")
        offset, payload_size = self._frame_offsets[index]
        with open(self.path, "rb") as fp:
            fp.seek(offset)
            payload = _read_exact(fp, payload_size)
        return _decode_frame_box_payload(payload)


def _encode_frame_header(frame: Dict[str, Any]) -> bytes:
    fhdr = {
        "frame_time": frame.get("frame_time"),
        "result_type": frame.get("result_type"),
        "motion_score": frame.get("motion_score"),
        "motion_roi": frame.get("motion_roi"),
        "inference_roi": frame.get("inference_roi"),
    }
    return _encode_value(fhdr)


def _encode_frame_box_payload(frame: Dict[str, Any]) -> bytes:
    parts: List[bytes] = []

    fhdr = _encode_frame_header(frame)
    parts.append(struct.pack(">I4s", 8 + len(fhdr), FOURCC_FRAME_HEADER) + fhdr)

    tracks = frame.get("objects")
    if tracks is not None:
        payload = _encode_value(encode_nested_arrays(tracks))
        parts.append(struct.pack(">I4s", 8 + len(payload), FOURCC_TRACKS) + payload)

    detections = frame.get("inference_dets")
    if detections is not None:
        payload = _encode_value(encode_nested_arrays(detections))
        parts.append(struct.pack(">I4s", 8 + len(payload), FOURCC_DETECTIONS) + payload)

    debug = frame.get("debug")
    if debug is not None:
        payload = _encode_value(encode_nested_arrays(debug))
        parts.append(struct.pack(">I4s", 8 + len(payload), FOURCC_DEBUG) + payload)

    image_path = frame.get("image_path")
    if image_path is not None:
        payload = str(image_path).encode("utf-8")
        parts.append(struct.pack(">I4s", 8 + len(payload), FOURCC_IMAGE_PATH) + payload)

    known_keys = {
        "frame_time",
        "result_type",
        "motion_score",
        "motion_roi",
        "inference_roi",
        "objects",
        "inference_dets",
        "debug",
        "image_path",
    }
    extras = {k: v for k, v in frame.items() if k not in known_keys}
    if extras:
        payload = _encode_value(encode_nested_arrays(extras))
        parts.append(struct.pack(">I4s", 8 + len(payload), FOURCC_EXTRA) + payload)

    return b"".join(parts)


def _decode_encoded_value(payload: bytes) -> Any:
    value, used = _decode_value(payload, 0)
    if used != len(payload):
        raise UBTRK2Error("Encoded value payload has trailing bytes")
    return value


def _decode_frame_box_payload(payload: bytes) -> Dict[str, Any]:
    frame: Dict[str, Any] = {
        "frame_time": None,
        "result_type": None,
        "motion_score": None,
        "motion_roi": None,
        "inference_roi": None,
        "objects": None,
        "image_path": None,
        "debug": None,
    }
    extras: Dict[str, Any] = {}
    for fourcc, child_payload in _iter_boxes(payload):
        if fourcc == FOURCC_FRAME_HEADER:
            fhdr = _decode_encoded_value(child_payload)
            if not isinstance(fhdr, dict):
                raise UBTRK2Error("fhdr payload must decode to a dict")
            frame["frame_time"] = fhdr.get("frame_time")
            frame["result_type"] = fhdr.get("result_type")
            frame["motion_score"] = fhdr.get("motion_score")
            frame["motion_roi"] = fhdr.get("motion_roi")
            frame["inference_roi"] = fhdr.get("inference_roi")
        elif fourcc == FOURCC_TRACKS:
            frame["objects"] = decode_nested_payloads(_decode_encoded_value(child_payload))
        elif fourcc == FOURCC_DETECTIONS:
            frame["inference_dets"] = decode_nested_payloads(
                _decode_encoded_value(child_payload)
            )
        elif fourcc == FOURCC_DEBUG:
            frame["debug"] = decode_nested_payloads(_decode_encoded_value(child_payload))
        elif fourcc == FOURCC_IMAGE_PATH:
            frame["image_path"] = child_payload.decode("utf-8")
        elif fourcc == FOURCC_EXTRA:
            decoded = decode_nested_payloads(_decode_encoded_value(child_payload))
            if isinstance(decoded, dict):
                extras.update(decoded)
    if extras:
        frame.update(extras)
    return frame

