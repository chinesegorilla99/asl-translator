import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import joblib
import json

WINDOW_NAME = "ASL Translator"
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

LANDMARK_STYLE = {
    "thumb": {"color": (255, 128, 0), "radius": 6},
    "index": {"color": (255, 0, 128), "radius": 6},
    "middle": {"color": (128, 0, 255), "radius": 6},
    "ring": {"color": (0, 128, 255), "radius": 6},
    "pinky": {"color": (0, 255, 128), "radius": 6},
    "wrist": {"color": (255, 255, 255), "radius": 7}
}

FINGER_INDICES = {
    "wrist": [0],
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}


class WebcamCapture:
    def __init__(self, camera_index=0, width=1280, height=720):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        return self

    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class HandTracker:
    def __init__(self, model_path=None, num_hands=2):
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "mediapipe" / "hand_landmarker.task"
        self.model_path = str(model_path)
        self.num_hands = num_hands
        self.detector = None

    def start(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        return self

    def detect(self, frame):
        if self.detector is None:
            return [], []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        
        if not result.hand_landmarks:
            return [], []
        
        all_coords = []
        all_landmarks = []
        
        for landmarks in result.hand_landmarks:
            coords = []
            for lm in landmarks:
                coords.extend([lm.x, lm.y, lm.z])
            all_coords.append(np.array(coords, dtype=np.float32))
            all_landmarks.append(landmarks)
        
        return all_coords, all_landmarks

    def stop(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    coords = landmarks.reshape(21, 3)
    wrist = coords[0].copy()
    centered = coords - wrist
    
    index_mcp = centered[5]
    pinky_mcp = centered[17]
    palm_width = np.linalg.norm(index_mcp - pinky_mcp)
    
    if palm_width > 0.001:
        normalized = centered / palm_width
    else:
        normalized = centered
    
    return normalized.flatten().astype(np.float32)


class PredictionSmoother:
    def __init__(self, window_size=10, min_confidence=0.5, min_agreement=0.5):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.history = []
    
    def add(self, prediction, confidence):
        if prediction is not None and confidence >= self.min_confidence:
            self.history.append((prediction, confidence))
        
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def get_smoothed(self):
        if not self.history:
            return None, 0.0
        
        votes = {}
        for pred, conf in self.history:
            if pred not in votes:
                votes[pred] = {"count": 0, "total_conf": 0.0}
            votes[pred]["count"] += 1
            votes[pred]["total_conf"] += conf
        
        best_pred = max(votes.keys(), key=lambda k: votes[k]["count"])
        best_count = votes[best_pred]["count"]
        best_avg_conf = votes[best_pred]["total_conf"] / best_count
        
        agreement = best_count / len(self.history)
        
        if agreement >= self.min_agreement:
            return best_pred, best_avg_conf
        else:
            return None, 0.0
    
    def clear(self):
        self.history = []


class ASLClassifier:
    def __init__(self, model_path=None, metadata_path=None):
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "saved" / "classifier.joblib"
        if metadata_path is None:
            metadata_path = Path(__file__).parent.parent.parent / "models" / "saved" / "model_metadata.json"
        
        self.model = joblib.load(model_path)
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        self.idx_to_label = {v: k for k, v in metadata["label_to_idx"].items()}
        self.normalized = metadata.get("normalized", False)

    def predict(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            return None, 0.0
        
        if self.normalized:
            landmarks = normalize_landmarks(landmarks)
        
        features = landmarks.reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_label = self.idx_to_label[predicted_idx].upper()
        
        return predicted_label, confidence


def draw_hand_skeleton(frame, landmarks):
    if landmarks is None:
        return frame
    
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    
    for finger_name, indices in FINGER_INDICES.items():
        style = LANDMARK_STYLE.get(finger_name, {"color": (255, 255, 255), "radius": 5})
        for idx in indices:
            cv2.circle(frame, points[idx], style["radius"], style["color"], -1)
            cv2.circle(frame, points[idx], style["radius"], (0, 0, 0), 1)
    
    return frame


CONFIDENCE_THRESHOLD = 0.90


def draw_prediction(frame, prediction, confidence):
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    if prediction is None or confidence < CONFIDENCE_THRESHOLD:
        display_text = "Unsure"
        color = (100, 100, 100)
        conf_text = f"Confidence: {confidence * 100:.0f}%" if confidence > 0 else ""
    else:
        display_text = prediction
        if confidence >= 0.95:
            color = (0, 255, 0)
        elif confidence >= 0.90:
            color = (0, 255, 255)
        else:
            color = (0, 100, 255)
        conf_text = f"Confidence: {confidence * 100:.0f}%"
    
    cv2.putText(frame, display_text, (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
    
    if conf_text:
        cv2.putText(frame, conf_text, (180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame


def draw_status_bar(frame, hand_detected):
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    if hand_detected:
        status_text = "Hand Detected - Show an ASL sign"
        status_color = (0, 255, 0)
    else:
        status_text = "No Hand Detected - Position your hand in frame"
        status_color = (100, 100, 100)
    
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, status_text, (text_x, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    cv2.putText(frame, "Press Q to quit", (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return frame


def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    classifier = ASLClassifier()
    smoother = PredictionSmoother(window_size=10, min_confidence=0.4, min_agreement=0.5)
    
    with WebcamCapture() as webcam, HandTracker(num_hands=1) as tracker:
        current_prediction = None
        current_confidence = 0.0
        
        while True:
            frame = webcam.read()
            if frame is None:
                break
            
            all_coords, all_landmarks = tracker.detect(frame)
            
            display_frame = cv2.flip(frame, 1)
            
            if all_landmarks:
                h, w = display_frame.shape[:2]
                for landmarks in all_landmarks:
                    points = [(w - 1 - int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                    for start, end in HAND_CONNECTIONS:
                        cv2.line(display_frame, points[start], points[end], (0, 255, 0), 2)
                    for finger_name, indices in FINGER_INDICES.items():
                        style = LANDMARK_STYLE.get(finger_name, {"color": (255, 255, 255), "radius": 5})
                        for idx in indices:
                            cv2.circle(display_frame, points[idx], style["radius"], style["color"], -1)
                            cv2.circle(display_frame, points[idx], style["radius"], (0, 0, 0), 1)
            
            hand_detected = len(all_landmarks) > 0
            
            if hand_detected and len(all_coords) > 0:
                raw_prediction, raw_confidence = classifier.predict(all_coords[0])
                smoother.add(raw_prediction, raw_confidence)
                current_prediction, current_confidence = smoother.get_smoothed()
            else:
                smoother.clear()
                current_prediction = None
                current_confidence = 0.0
            
            display_frame = draw_prediction(display_frame, current_prediction, current_confidence)
            display_frame = draw_status_bar(display_frame, hand_detected)
            
            cv2.imshow(WINDOW_NAME, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
