"""CLIP/MobileCLIP text encoding, JPEG embedding pipeline, and cosine similarity / softmax."""
import math
import time

import stuff.coord as coord


def clip_encode_text(text_strings, model_choice="mobileclip_s0", device="cuda"):
    """
    Encode text with MobileCLIP v1 (mobileclip) or MobileCLIP2 (open_clip).
    Returns L2-normalized features as Python lists.
    """
    import os
    import torch
    import numpy as np
    from contextlib import nullcontext

    # Decide v1 vs v2 by name
    is_v2 = ("mobileclip2" in model_choice.lower()) or model_choice.startswith("MobileCLIP2")

    # Checkpoint path convention (adjust if yours differs)
    ckpt_name = (model_choice.lower().replace("-", "_") if is_v2 else model_choice) + ".pt"
    ckpt_path = os.path.join("/mldata/models/clip/pt", ckpt_name)

    # Load model & tokenizer
    if is_v2:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(model_choice, pretrained=ckpt_path)
        tokenizer = open_clip.get_tokenizer(model_choice)
    else:
        import mobileclip
        model, _, _ = mobileclip.create_model_and_transforms(model_choice, pretrained=ckpt_path)
        tokenizer = mobileclip.get_tokenizer(model_choice)

    # Optional/safe reparam (won't crash if unsupported)
    def _safe_reparam(m):
        try:
            from mobileclip.modules.common.mobileone import reparameterize_model as _rp
            # Only attempt if there is at least one submodule with .reparameterize
            if any(hasattr(s, "reparameterize") for s in m.modules()):
                try:
                    return _rp(m)
                except Exception as e:
                    print(f"⚠️ Skipping reparameterize_model: {e}")
                    return m
            return m
        except Exception:
            return m

    model = _safe_reparam(model.eval())
    model = model.to(device)

    # --- Robust tokenization handling ---
    tokens = tokenizer(text_strings)

    # Handle dict-like (e.g., some tokenizers) -> prefer "input_ids"
    if isinstance(tokens, dict):
        tokens = tokens.get("input_ids", next(iter(tokens.values())))

    # Convert numpy / list to tensor
    if isinstance(tokens, (list, tuple)):
        tokens = np.array(tokens)

    if isinstance(tokens, np.ndarray):
        # Ensure 2D: [batch, seq]
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        tokens = torch.from_numpy(tokens)

    # Ensure torch.Tensor, int64 indices, and right device
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens)

    tokens = tokens.to(dtype=torch.long, device=device)

    # --- Encode (no autocast; embeddings use integer indices) ---
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.detach().cpu().tolist()

def get_jpeg_embeddings(upyc_cli, jpegs, track_shared=None):
    # analyse a list of jpegs (as python byte arrays), get face, clip
    #  embeddings and jpegs for the largest person detection
    # in the image. Also return the whole image clip embedding
    if track_shared is None:
        track_shared=upyc_cli.c_track_shared_state("/mldata/config/track/trackers/uc_v9.yaml")
    track_stream=upyc_cli.c_track_stream(track_shared)

    for j in jpegs:
        track_stream.run_on_jpeg(j, True, 0)
    #print(len(jpegs))
    #time.sleep(1)
    res=track_stream.get_results(True)
    #print(len(res),"!")

    all_ret=[]
    for track_res in res:
        ret={}
        r=track_res["track_dets"]
        largest_area=0
        best_det=None
        for det in r:
            a=coord.box_a(det["box"])
            if a>largest_area and "face_embedding" in det and "face_jpeg" in det:
                largest_area=a
                best_det=det
        if best_det is not None:
            ret["face_embedding"]=best_det["face_embedding"]
            ret["face_jpeg"]=best_det["face_jpeg"]
            if "clip_embedding" in best_det:
                ret["person_clip_embedding"]=best_det["clip_embedding"]
            if "clip_jpeg" in best_det:
                ret["person_clip_jpeg"]=best_det["clip_jpeg"]
        if "frame_jpeg" in track_res:
            ret["frame_jpeg"] = track_res["frame_jpeg"]
        if "clip_embedding" in track_res:
            ret["frame_clip_embedding"] = track_res["clip_embedding"]
        all_ret.append(ret)
    return all_ret

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be the same length")
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        raise ValueError("One of the vectors is zero-length")
    return dot / (norm1 * norm2)

def cosine_similarities_to_probabilities(input, refs, scale=100.0):

    ds=[]
    for embedding in refs:
        ds.append(cosine_similarity(input, embedding))

    # Scale the cosine similarities to logits
    logits = [scale * s for s in ds]

    # Numerically stable softmax
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)

    # Convert to probabilities
    probs = [e / sum_exps for e in exps]
    return probs
