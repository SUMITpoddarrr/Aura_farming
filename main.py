import cv2
import time
import math
import numpy as np
from mediapipe.python.solutions import selfie_segmentation as mp_selfie
from mediapipe.python.solutions import face_detection as mp_face

from utils.config import *
from detectors.emotion_detector import EmotionDetector
from graphics.aura_effect import create_edge_aura_layer, additive_blend
from utils.video_utils import draw_particles


def main():
    detector = EmotionDetector(mode="deepface")
    print(">>> Emotion detector starting in mode:", detector.mode)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        seg_res = selfie.process(rgb)
        # Avoid numpy truth-value ambiguity; explicitly check for None
        mask_prob = seg_res.segmentation_mask if seg_res.segmentation_mask is not None else np.ones((HEIGHT, WIDTH), np.float32)
        mask_smooth = cv2.GaussianBlur(mask_prob, (MASK_BLUR, MASK_BLUR), 0)

        face_res = face_det.process(rgb)
        face_crop = None
        if face_res.detections:
            bbox = face_res.detections[0].location_data.relative_bounding_box
            x = int(bbox.xmin * WIDTH)
            y = int(bbox.ymin * HEIGHT)
            bw, bh = int(bbox.width * WIDTH), int(bbox.height * HEIGHT)
            pad = int(max(bw, bh) * 0.25)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(WIDTH, x + bw + pad), min(HEIGHT, y + bh + pad)
            face_crop = frame[y1:y2, x1:x2].copy()

        tnow = time.time()
        if tnow - timestamp > 0.8:
            timestamp = tnow
            emotion_label, emotion_conf = detector.detect(face_crop)

        dt = max(1e-3, tnow - last_time)
        last_time = tnow
        color_alpha = 1.0 - math.exp(-6 * dt)
        conf_alpha = 1.0 - math.exp(-4 * dt)
        target_color = np.array(EMOTION_COLOR.get(emotion_label,
                                  EMOTION_COLOR["neutral"]), np.float32)
        current_color = (1 - color_alpha) * current_color + color_alpha * target_color
        smoothed_conf = (1 - conf_alpha) * smoothed_conf + conf_alpha * emotion_conf

        boost = EMO_BOOST.get(emotion_label, 1.0)
        dynamic_exp = int(glow_size + (110 * smoothed_conf * boost))
        dynamic_int = np.clip(BASE_INTENSITY * 0.7 +
                              1.1 * smoothed_conf * boost, 0.25, 1.8)

        aura = create_edge_aura_layer(mask_smooth,
                                      tuple(current_color.astype(np.uint8)),
                                      dynamic_int, dynamic_exp, AURA_BLUR, tnow)

        pulse = 0.5 * (1 + math.sin(2 * math.pi * PULSE_FREQ * (tnow % 10)))
        pulse_alpha = 0.25 + 0.75 * pulse * min(1.0, smoothed_conf + 0.2)
        blended = additive_blend(frame, aura, alpha=pulse_alpha)

        particles = draw_particles(mask_smooth,
                                   tuple(current_color.astype(np.uint8)),
                                   smoothed_conf)
        blended = additive_blend(blended, particles, alpha=0.5 * smoothed_conf)

        txt = f"{emotion_label} ({smoothed_conf:.2f})"
        cv2.putText(blended, txt, (10, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.imshow("Aura Farming", blended)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Disable mode toggling; DeepFace-only
        # elif key == ord('m'):
        #     detector.set_mode("deepface")
        elif key == ord('s'):
            cv2.imwrite("aura_capture.png", blended)
        elif key in (ord('+'), ord('=')):
            glow_size = min(glow_size + 8, 200)
        elif key in (ord('-'), ord('_')):
            glow_size = max(8, glow_size - 8)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
