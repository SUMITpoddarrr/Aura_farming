import cv2
import numpy as np
import time
import argparse
import math


USE_DEEPFACE_AVAILABLE = False
try:
    from fer import FER
    fer_detector = FER(mtcnn=True) 
except Exception as e:
    fer_detector = None
    print("FER not available or failed to init:", e)

try:
    from deepface import DeepFace
    USE_DEEPFACE_AVAILABLE = True
except Exception:
    USE_DEEPFACE_AVAILABLE = False

# Import MediaPipe solutions directly to avoid importing TensorFlow via mediapipe.tasks
from mediapipe.python.solutions import selfie_segmentation as mp_selfie
from mediapipe.python.solutions import face_detection as mp_face


WIDTH, HEIGHT = 640, 480
MASK_BLUR = 21         
AURA_EXPANSION = 40   
AURA_BLUR = 55         
BASE_INTENSITY = 0.6   
PULSE_FREQ = 1.2       



EMOTION_COLOR = {
    "angry":    (0, 0, 220),    # red-ish
    "disgust":  (0, 150, 50),   # green
    "fear":     (120, 0, 120),  # purple
    "happy":    (0, 200, 220),  # yellowish-cyan-ish 
    "sad":      (200, 80, 0),   # blue-ish 
    "surprise": (0, 255, 255),  # cyan
    "neutral":  (200, 120, 200) # soft magenta
}


def normalise_emotion_label(label: str) -> str:
    label = label.lower()
    if label in EMOTION_COLOR:
        return label
    if label in ("happy", "happiness"):
        return "happy"
    if label in ("sad", "sadness"):
        return "sad"
    if label in ("surprise", "surprised"):
        return "surprise"
    if label in ("neutral", "neutrality"):
        return "neutral"
    if "angry" in label or "anger" in label:
        return "angry"
    if "disgust" in label or "disliked" in label:
        return "disgust"
    if "fear" in label or "scared" in label:
        return "fear"
    return "neutral"


class EmotionDetector:
    def __init__(self, mode="fer"):
        self.mode = mode
        self.fer = fer_detector if fer_detector is not None else None
        self.use_deepface = False
        if USE_DEEPFACE_AVAILABLE:
            self.deepface_available = True
        else:
            self.deepface_available = False

    def set_mode(self, mode):
        if mode == "deepface" and not self.deepface_available:
            print("DeepFace not available. Staying on FER.")
            return
        self.mode = mode

    def detect(self, face_img):
        # defensive
        if face_img is None or face_img.size == 0:
            print("[DEBUG] Empty face crop")
            return ("neutral", 0.0)
        print("Face crop size:", face_img.shape)
        
        if self.mode == "fer" and self.fer is not None:
            try:
                rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                results = self.fer.top_emotion(rgb)  # (label, score)
                if results is None:
                    return ("neutral", 0.0)
                label, score = results
                label = normalise_emotion_label(label)
                return (label, float(score))
            except Exception as e:
                return ("neutral", 0.0)

        # DeepFace 
        if self.mode == "deepface" and self.deepface_available:
            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                res = DeepFace.analyze(rgb_face,actions=['emotion'],enforce_detection=False)
                #print("DeepFace raw result:", res)
               
                if isinstance(res, list) and len(res) > 0:
                    res = res[0]
                emos = {}
                dominant = None
                if isinstance(res, dict):
                    if 'emotion' in res and isinstance(res['emotion'], dict):
                        emos = res['emotion']
                    elif 'emotions' in res and isinstance(res['emotions'], dict):
                        emos = res['emotions']
                    dominant = res.get('dominant_emotion')
                #print("DeepFace parsed emotions:", emos, "dominant:", dominant)
                if not emos and dominant:
                    return (normalise_emotion_label(dominant), 0.6)
                if not emos:
                    print("DeepFace returned no emotions")
                    return ("neutral", 0.0)
                label, score = max(emos.items(), key=lambda x: x[1])
                score_f = float(score)
                if score_f > 1.0:
                    score_f = score_f / 100.0
                score_f = max(0.0, min(1.0, score_f))
                return (normalise_emotion_label(label), score_f)
            except Exception as e:
                print("DeepFace error:", e)
                return ("neutral", 0.0)

        return ("neutral", 0.0)


def create_edge_aura_layer(mask, color_bgr, intensity=0.7, expansion=AURA_EXPANSION, blur=AURA_BLUR, phase: float = 0.0):
    h, w = mask.shape
    mask_bin = (mask > 0.3).astype(np.uint8) * 255
    outside = 255 - mask_bin
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3)  

    # Map distance to tapered alpha: bright thin inner rim, softly fading outer halo
    expansion_px = max(8.0, float(expansion))
    d_norm = np.clip(dist / expansion_px, 0.0, 1.0)
    # Inner rim (sharp near body): Gaussian falloff in first ~12 px
    rim_sigma = 10.0
    rim = np.exp(- (dist / rim_sigma) ** 2)
    # Outer halo: linear-to-quadratic falloff to keep edges lighter
    halo = (1.0 - d_norm)
    halo = np.clip(halo, 0.0, 1.0) ** 2.0
    # Combine and mask to only outside region
    ring_f = (0.65 * rim + 0.55 * halo)
    ring_f = np.clip(ring_f, 0.0, 1.0) * (outside > 0).astype(np.float32)

    # Subtle flow animation along the band
    yy, xx = np.indices((h, w))
    flow = 0.75 + 0.25 * np.sin(0.045 * xx + 0.065 * yy + phase * 1.6)
    ring_f = np.clip(ring_f * flow.astype(np.float32), 0.0, 1.0)

    # Colorize
    col = np.zeros((h, w, 3), dtype=np.float32)
    col[:, :, 0] = color_bgr[0] / 255.0
    col[:, :, 1] = color_bgr[1] / 255.0
    col[:, :, 2] = color_bgr[2] / 255.0
    aura = col * np.expand_dims(ring_f, axis=2)

    # Two-stage blur: small to soften rim, larger to extend halo while keeping tapered edge
    k_small = int(max(1, (blur // 3)) // 2 * 2 + 1)
    k_large = int(max(3, blur) // 2 * 2 + 1)
    soft = cv2.GaussianBlur(aura, (k_small, k_small), 0)
    wide = cv2.GaussianBlur(aura, (k_large, k_large), 0)
    aura_blur = np.clip(0.7 * soft + 0.6 * wide, 0, 1)
    aura_blur = np.clip(aura_blur * float(intensity), 0, 1)
    return (aura_blur * 255).astype(np.uint8)


def additive_blend(background, aura_layer, alpha=0.7):
    bg_f = background.astype(np.float32) / 255.0
    aura_f = aura_layer.astype(np.float32) / 255.0
    out = bg_f + aura_f * alpha
    out = np.clip(out, 0, 1.0)
    return (out * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fer", "deepface"], default="fer",
                        help="emotion detection mode (fer faster, deepface optional slower)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    detector = EmotionDetector(mode=args.mode)
    print(">>> Emotion detector starting in mode:", detector.mode)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    # MediaPipe instances: selfie seg + face detection
    selfie = mp_selfie.SelfieSegmentation(model_selection=1)
    face_det = mp_face.FaceDetection(min_detection_confidence=0.5)

    glow_size = AURA_EXPANSION
    last_time = time.time()
    timestamp = last_time
    emotion_label = "neutral"
    emotion_conf = 0.0
    pulse_phase = 0.0

    current_color = np.array(EMOTION_COLOR["neutral"], dtype=np.float32)
    smoothed_conf = 0.0
    COLOR_SMOOTH_RATE = 6.0    # higher = faster transitions (per second)
    CONF_SMOOTH_RATE = 4.0     # higher = faster confidence changes (per second)

    # Per-emotion boost for intensity/area
    EMO_BOOST = {
        "angry": 1.35,
        "fear": 1.15,
        "surprise": 1.10,
        "happy": 1.05,
        "disgust": 1.05,
        "sad": 0.95,
        "neutral": 0.85,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (WIDTH, HEIGHT))

        # segmentation mask
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        seg_res = selfie.process(rgb)
        mask_prob = seg_res.segmentation_mask if seg_res.segmentation_mask is not None else np.ones((HEIGHT, WIDTH), dtype=np.float32)
        mask_smooth = cv2.GaussianBlur(mask_prob, (MASK_BLUR, MASK_BLUR), 0)
        mask_bin = (mask_smooth > 0.3).astype(np.uint8)

        # face detection (MediaPipe)
        face_rgb = rgb.copy()
        face_res = face_det.process(face_rgb)
        face_bbox = None
        face_crop = None
        if face_res.detections:
            # Use first detection
            det = face_res.detections[0]
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * WIDTH)
            y = int(bbox.ymin * HEIGHT)
            bw = int(bbox.width * WIDTH)
            bh = int(bbox.height * HEIGHT)
            # expand bbox a bit
            pad = int(max(bw, bh) * 0.25)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(WIDTH, x + bw + pad)
            y2 = min(HEIGHT, y + bh + pad)
            face_bbox = (x1, y1, x2, y2)
            face_crop = small[y1:y2, x1:x2].copy()
        else:
            # fallback
            cx, cy = WIDTH // 2, HEIGHT // 3
            wbox = WIDTH // 4
            hbox = HEIGHT // 4
            x1 = max(0, cx - wbox)
            y1 = max(0, cy - hbox)
            x2 = min(WIDTH, cx + wbox)
            y2 = min(HEIGHT, cy + hbox)
            face_bbox = (x1, y1, x2, y2)
            face_crop = small[y1:y2, x1:x2].copy()
        # if face_bbox:
        #     x1, y1, x2, y2 = face_bbox
        #     cv2.rectangle(small, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Emotion detection periodically
        tnow = time.time()
        if tnow - timestamp > 0.8:  
            timestamp = tnow
            label, conf = detector.detect(face_crop)
            emotion_label = label
            emotion_conf = conf

        # Choose target color and smooth transitions over time
        target_color = np.array(EMOTION_COLOR.get(emotion_label, EMOTION_COLOR["neutral"]), dtype=np.float32)
        dt = max(1e-3, tnow - last_time)
        color_alpha = 1.0 - math.exp(-COLOR_SMOOTH_RATE * dt)
        conf_alpha  = 1.0 - math.exp(-CONF_SMOOTH_RATE * dt)
        current_color = (1.0 - color_alpha) * current_color + color_alpha * target_color
        smoothed_conf = (1.0 - conf_alpha) * smoothed_conf + conf_alpha * float(emotion_conf)

        # Per-emotion boost
        emo_boost = EMO_BOOST.get(emotion_label, 1.0)

        # Dynamic area and intensity scale with confidence
        dynamic_expansion = int(glow_size + (110 * smoothed_conf * emo_boost))
        dynamic_intensity = np.clip(BASE_INTENSITY * 0.7 + 1.1 * smoothed_conf * emo_boost, 0.25, 1.8)

        # create edge-only aura around silhouette with flowing band
        phase = (tnow)  # time-based phase
        aura = create_edge_aura_layer(
            mask_smooth,
            color_bgr=tuple(current_color.astype(np.uint8).tolist()),
            intensity=dynamic_intensity,
            expansion=dynamic_expansion,
            blur=AURA_BLUR,
            phase=phase
        )

        # pulse using sin wave modulated by smoothed confidence
        pulse = 0.5 * (1.0 + math.sin(2 * math.pi * PULSE_FREQ * (tnow % 10)))  # 0..1
        pulse_alpha = 0.25 + 0.75 * pulse * min(1.0, smoothed_conf + 0.2)

        blended = additive_blend(small, aura, alpha=pulse_alpha)

        tinted = blended

        # draw subtle particles around silhouette for excitement/energy
        particle_layer = np.zeros_like(tinted)
        if emotion_conf > 0.15:
            # simple stationary particles seeded around the mask edges
            edges = cv2.Canny((mask_smooth * 255).astype(np.uint8), 50, 150)
            ys, xs = np.where(edges > 0)
            n = min(120, len(xs))
            if n > 0:
                idx = np.random.choice(range(len(xs)), n, replace=False)
                base_color = tuple(current_color.astype(np.uint8).tolist())
                for i in idx:
                    px, py = xs[i], ys[i]
                    rad = np.random.randint(1, 4)
                    col = (int(np.clip(base_color[0] + np.random.randint(-30, 30), 0, 255)),
                           int(np.clip(base_color[1] + np.random.randint(-30, 30), 0, 255)),
                           int(np.clip(base_color[2] + np.random.randint(-30, 30), 0, 255)))
                    cv2.circle(particle_layer, (px, py), rad, col, -1)
                particle_layer = cv2.GaussianBlur(particle_layer, (7, 7), 0)
                tinted = additive_blend(tinted, particle_layer, alpha=0.5 * smoothed_conf)

        txt = f"{emotion_label} ({smoothed_conf:.2f})"
        cv2.putText(tinted, txt, (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2, cv2.LINE_AA)

        cv2.imshow("Aura Farming", tinted)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            if detector.mode == "fer" and detector.deepface_available:
                detector.set_mode("deepface")
                print("Switched to DeepFace (slower).")
            else:
                detector.set_mode("fer")
                print("Switched to FER (faster).")
        elif key == ord('s'):
            cv2.imwrite("aura_capture.png", tinted)
            print("Saved aura_capture.png")
        elif key == ord('+') or key == ord('='):
            glow_size = min(glow_size + 8, 200)
            print("Glow size:", glow_size)
        elif key == ord('-') or key == ord('_'):
            glow_size = max(8, glow_size - 8)
            print("Glow size:", glow_size)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
