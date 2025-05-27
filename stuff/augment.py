import cv2
import numpy as np

def bt709_yuv420_augment(
    batch_bgr: np.ndarray,
    randomize: bool = False,
    coeff_range: float = 0.05,
    subsample_error_prob: float = 0.1,
    subsample_error_range: int = 2,
) -> np.ndarray:
    """
    Simulate BT.709 YUV420 round-trip with optional randomized error.

    Args:
        batch_bgr (np.ndarray): (N, H, W, 3) uint8 BGR images.
        randomize (bool): If True, apply small random perturbations.
        coeff_range (float): Max relative jitter on YUV conversion coeffs (e.g. 0.05 for ±5%).
        subsample_error_prob (float): Probability per image of a chroma-dim error.
        subsample_error_range (int): Max absolute pixel offset in [-R..+R] for chroma dims.

    Returns:
        np.ndarray: (N, H, W, 3) BGR uint8, after lossy YUV420 round-trip.
    """
    # Nominal BT.709 coefficients
    Kr0, Kg0, Kb0 = 0.2126, 0.7152, 0.0722
    inv0 = {
        'u_scale': 2.12798,
        'v_scale': 1.28033,
        'g_u': -0.21482,
        'g_v': -0.38059,
    }

    N, H, W, _ = batch_bgr.shape
    out = np.empty_like(batch_bgr)
    rng = np.random.default_rng()

    for i in range(N):
        img = batch_bgr[i].astype(np.float32)
        B, G, R = cv2.split(img)

        # --- potentially jitter coefficients ---
        if randomize:
            # sample multiplicative factors in [1-coeff_range, 1+coeff_range]
            f = rng.uniform(1 - coeff_range, 1 + coeff_range, size=3)
            Kr, Kg, Kb = Kr0 * f[0], Kg0 * f[1], Kb0 * f[2]
            # jitter inverse scales similarly
            inv = {
                k: v * rng.uniform(1 - coeff_range, 1 + coeff_range)
                for k, v in inv0.items()
            }
        else:
            Kr, Kg, Kb = Kr0, Kg0, Kb0
            inv = inv0

        # 1) to YUV
        Y = Kr*R + Kg*G + Kb*B
        U = (B - Y) * 0.5389 + 128.0
        V = (R - Y) * 0.6350 + 128.0

        # --- determine chroma sampling dims ---
        base_w, base_h = W // 2, H // 2
        if randomize and rng.random() < subsample_error_prob:
            # apply independent random integer offsets in [-subsample_error_range..+]
            dw = int(rng.integers(-subsample_error_range, subsample_error_range + 1))
            dh = int(rng.integers(-subsample_error_range, subsample_error_range + 1))
            sw = max(1, base_w + dw)
            sh = max(1, base_h + dh)
        else:
            sw, sh = base_w, base_h

        # 2) downsample & upsample U,V
        U_ds = cv2.resize(U, (sw, sh), interpolation=cv2.INTER_AREA)
        V_ds = cv2.resize(V, (sw, sh), interpolation=cv2.INTER_AREA)
        U_us = cv2.resize(U_ds, (W, H), interpolation=cv2.INTER_NEAREST)
        V_us = cv2.resize(V_ds, (W, H), interpolation=cv2.INTER_NEAREST)

        # 3) back to BGR
        Um = U_us - 128.0
        Vm = V_us - 128.0
        Rr = Y + inv['v_scale'] * Vm
        Bb = Y + inv['u_scale'] * Um
        Gg = Y + inv['g_u'] * Um + inv['g_v'] * Vm

        # 4) clip & store
        out[i, :, :, 0] = np.clip(Bb, 0, 255).astype(np.uint8)
        out[i, :, :, 1] = np.clip(Gg, 0, 255).astype(np.uint8)
        out[i, :, :, 2] = np.clip(Rr, 0, 255).astype(np.uint8)

    return out

def bt709_yuv420_augment_single(img: np.ndarray) -> np.ndarray:
    return bt709_yuv420_augment(np.expand_dims(img, axis=0))[0]