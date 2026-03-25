from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np
import os
import json
import pickle

def phash_to_bytes(img: Image.Image) -> np.ndarray:
    """Compute 64-bit pHash and return packed uint8[8] for FAISS binary index."""
    import imagehash
    h = imagehash.phash(img)           # 8x8 boolean-ish array under the hood
    bits = np.array(h.hash, dtype=np.uint8).reshape(-1)   # shape (64,)
    return np.packbits(bits)[None, :]  # shape (1, 8); FAISS expects (n, d/8)

@dataclass
class MatchResult:
    duplicate: bool
    distance: Optional[int] = None         # Hamming distance (0..64)
    match_id: Optional[int] = None         # index into state.paths
    match_path: Optional[str] = None

class ImgDedup:
    """
    Streaming image deduper using 64-bit perceptual hash + FAISS binary index (Hamming).
    - test(image_path) -> MatchResult (and adds to index/state)
    - save/load for persistence
    """
    def __init__(self, threshold: int = 5):
        import faiss
        """
        threshold: maximum Hamming distance to consider 'duplicate'.
                   0–5 works well for scaled/re-encoded JPEGs.
        """
        self.threshold = int(threshold)
        self.dim_bits = 64
        self.index = faiss.IndexBinaryFlat(self.dim_bits)  # Hamming distance
        self.paths: list[str] = []                         # parallel array of metadata

    @property
    def size(self) -> int:
        return self.index.ntotal

    def _vectorize_path(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            return phash_to_bytes(im)  # (1, 8)

    def test(self, image_path: str) -> MatchResult:
        """
        Check if image is a near-duplicate of anything seen before and add it to the state.
        Returns a MatchResult with duplicate flag, best distance, and matched path (if any).
        """
        xq = self._vectorize_path(image_path)  # (1, 8)

        if self.size > 0:
            # k=1: only need best hit; distances are Hamming (int32)
            D, I = self.index.search(xq, k=1)
            best_dist = int(D[0, 0])
            best_id = int(I[0, 0])
            is_dup = best_dist <= self.threshold
        else:
            best_dist, best_id, is_dup = None, None, False

        # Always add to state (so future calls see it)
        self.index.add(xq)
        self.paths.append(os.path.abspath(image_path))

        if is_dup:
            return MatchResult(
                duplicate=True,
                distance=best_dist,
                match_id=best_id,
                match_path=self.paths[best_id] if best_id is not None and best_id >= 0 else None,
            )
        else:
            return MatchResult(duplicate=False, distance=best_dist)

    # -------- Optional: convenience for batch processing --------
    def test_many(self, image_paths: list[str]) -> list[MatchResult]:
        results = []
        for p in image_paths:
            try:
                results.append(self.test(p))
            except Exception as e:
                results.append(MatchResult(duplicate=False, distance=None, match_path=f"ERROR: {e}"))
        return results

    # -------- Optional: persistence (save/load) --------
    def save(self, folder: str) -> None:
        import faiss
        os.makedirs(folder, exist_ok=True)
        faiss.write_index_binary(self.index, os.path.join(folder, "index.faissbin"))
        with open(os.path.join(folder, "paths.pkl"), "wb") as f:
            pickle.dump(self.paths, f)
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump({"threshold": self.threshold, "dim_bits": self.dim_bits}, f)

    @classmethod
    def load(cls, folder: str) -> "ImgDedup":
        import faiss
        with open(os.path.join(folder, "meta.json"), "r") as f:
            meta = json.load(f)
        state = cls(threshold=int(meta["threshold"]))
        state.index = faiss.read_index_binary(os.path.join(folder, "index.faissbin"))
        with open(os.path.join(folder, "paths.pkl"), "rb") as f:
            state.paths = pickle.load(f)
        return state