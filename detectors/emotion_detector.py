import cv2
from utils.config import EMOTION_COLOR

# Force DeepFace-only mode. We purposely do not initialize FER anymore.
USE_DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    USE_DEEPFACE_AVAILABLE = True
except Exception as _e:
    USE_DEEPFACE_AVAILABLE = False


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
    if "disgust" in label:
        return "disgust"
    if "fear" in label or "scared" in label:
        return "fear"
    return "neutral"


class EmotionDetector:
    def __init__(self, mode="deepface"):
        # Always use DeepFace regardless of the passed mode
        self.mode = "deepface"
        self.deepface_available = USE_DEEPFACE_AVAILABLE

    def set_mode(self, mode):
        # DeepFace-only; ignore attempts to change mode
        if mode != "deepface":
            print("DeepFace-only mode is enforced. Ignoring mode change.")
        self.mode = "deepface"

    def detect(self, face_img):
        if face_img is None or face_img.size == 0:
            return ("neutral", 0.0)

        # DeepFace-only path
        if self.deepface_available:
            try:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                res = DeepFace.analyze(rgb_face, actions=['emotion'],
                                       enforce_detection=False)
                if isinstance(res, list) and len(res) > 0:
                    res = res[0]
                emos = res.get('emotion') or res.get('emotions', {})
                dominant = res.get('dominant_emotion')
                if not emos and dominant:
                    return (normalise_emotion_label(dominant), 0.6)
                if not emos:
                    return ("neutral", 0.0)
                label, score = max(emos.items(), key=lambda x: x[1])
                score_f = float(score)
                if score_f > 1.0:
                    score_f /= 100.0
                return (normalise_emotion_label(label),
                        max(0.0, min(1.0, score_f)))
            except Exception as e:
                print("DeepFace error:", e)
                return ("neutral", 0.0)

        return ("neutral", 0.0)
