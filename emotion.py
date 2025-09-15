import cv2
from deepface import DeepFace
from mediapipe.python.solutions import face_detection as mp_face

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_det = mp_face.FaceDetection(min_detection_confidence=0.5)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res = face_det.process(rgb)
    if res.detections:
        det = res.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        # Draw bounding box
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            try:
                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                result = DeepFace.analyze(
                    rgb_face,
                    actions=["emotion"],
                    enforce_detection=False
                )
                if isinstance(result, list):
                    result = result[0]

                emotions = result.get("emotion", {})
                if emotions:
                    top_emotion = max(emotions, key=emotions.get)
                    conf = emotions[top_emotion]
                    txt = f"{top_emotion} ({conf:.1f}%)"
                    print(txt)
                    cv2.putText(frame, txt, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print("DeepFace error:", e)

    cv2.imshow("DeepFace + MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
