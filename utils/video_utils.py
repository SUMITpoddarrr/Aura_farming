import cv2
import numpy as np

def draw_particles(mask_smooth, base_color, conf):
    layer = np.zeros((mask_smooth.shape[0], mask_smooth.shape[1], 3),
                     dtype=np.uint8)
    if conf > 0.15:
        edges = cv2.Canny((mask_smooth * 255).astype(np.uint8), 50, 150)
        ys, xs = np.where(edges > 0)
        n = min(120, len(xs))
        if n:
            idx = np.random.choice(range(len(xs)), n, replace=False)
            for i in idx:
                px, py = xs[i], ys[i]
                rad = np.random.randint(1, 4)
                # Avoid uint8 overflow/underflow by casting to int before jitter
                b, g, r = int(base_color[0]), int(base_color[1]), int(base_color[2])
                jb = int(np.clip(b + int(np.random.randint(-30, 30)), 0, 255))
                jg = int(np.clip(g + int(np.random.randint(-30, 30)), 0, 255))
                jr = int(np.clip(r + int(np.random.randint(-30, 30)), 0, 255))
                col = (jb, jg, jr)
                cv2.circle(layer, (px, py), rad, col, -1)
            layer = cv2.GaussianBlur(layer, (7, 7), 0)
    return layer
