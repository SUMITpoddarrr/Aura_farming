import cv2
import numpy as np

def create_edge_aura_layer(mask, color_bgr, intensity, expansion, blur, phase):
    h, w = mask.shape
    mask_bin = (mask > 0.3).astype(np.uint8) * 255
    outside = 255 - mask_bin
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3)

    expansion_px = max(8.0, float(expansion))
    d_norm = np.clip(dist / expansion_px, 0.0, 1.0)
    rim_sigma = 10.0
    rim = np.exp(-(dist / rim_sigma) ** 2)
    halo = np.clip((1.0 - d_norm), 0.0, 1.0) ** 2
    ring_f = np.clip((0.65 * rim + 0.55 * halo), 0.0, 1.0) * (outside > 0)

    yy, xx = np.indices((h, w))
    flow = 0.75 + 0.25 * np.sin(0.045 * xx + 0.065 * yy + phase * 1.6)
    ring_f = np.clip(ring_f * flow.astype(np.float32), 0.0, 1.0)

    col = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        col[:, :, i] = color_bgr[i] / 255.0
    aura = col * ring_f[..., None]

    k_small = int(max(1, (blur // 3)) // 2 * 2 + 1)
    k_large = int(max(3, blur) // 2 * 2 + 1)
    soft = cv2.GaussianBlur(aura, (k_small, k_small), 0)
    wide = cv2.GaussianBlur(aura, (k_large, k_large), 0)
    aura_blur = np.clip(0.7 * soft + 0.6 * wide, 0, 1)
    return (aura_blur * float(intensity) * 255).astype(np.uint8)


def additive_blend(background, aura_layer, alpha=0.7):
    bg_f = background.astype(np.float32) / 255.0
    aura_f = aura_layer.astype(np.float32) / 255.0
    return (np.clip(bg_f + aura_f * alpha, 0, 1) * 255).astype(np.uint8)
