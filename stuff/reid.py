from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch

def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Computes the cosine similarity between two 1D tensors.

    Args:
        vec1 (torch.Tensor): First feature vector of shape [256], float32 or float16.
        vec2 (torch.Tensor): Second feature vector of shape [256], float32 or float16.

    Returns:
        float: Cosine similarity between vec1 and vec2.
    """
    import torch
    import torch.nn.functional as F

    if vec1 is None or vec2 is None:
        return 0

    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape")
    if vec1.dim() != 1:
        raise ValueError("Both inputs must be 1D tensors")

    # Optionally cast to float32 for numerical stability
    if vec1.dtype == torch.float16 or vec2.dtype == torch.float16:
        vec1 = vec1.to(torch.float32)
        vec2 = vec2.to(torch.float32)

    # Normalize and compute dot product
    return F.cosine_similarity(vec1, vec2, dim=0).item()