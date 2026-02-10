import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

from webcam_capture import WebcamCapture

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

LANDMARK_STYLE = {
    "default": {"color": (255, 255, 255), "radius": 5},
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


class RealtimeLandmarkExtractor:
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
        print(f"Hand landmarker loaded (tracking up to {self.num_hands} hands)")
        return self

    def extract(self, frame):
        if self.detector is None:
            raise RuntimeError("Extractor not started. Call start() first.")
        
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
            print("Hand landmarker closed")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def draw_landmarks(frame, landmarks):
    if landmarks is None:
        return frame
    
    h, w = frame.shape[:2]
    
    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
    
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    
    for finger_name, indices in FINGER_INDICES.items():
        style = LANDMARK_STYLE.get(finger_name, LANDMARK_STYLE["default"])
        for idx in indices:
            cv2.circle(frame, points[idx], style["radius"], style["color"], -1)
            cv2.circle(frame, points[idx], style["radius"], (0, 0, 0), 1)
    
    return frame


def main():
    print("Starting real-time landmark extraction...")
    print("Press 'q' to quit")
    
    with WebcamCapture() as webcam, RealtimeLandmarkExtractor(num_hands=2) as extractor:
        detected_count = 0
        total_count = 0
        
        while True:
            frame = webcam.read()
            if frame is None:
                break
            
            total_count += 1
            all_coords, all_landmarks = extractor.extract(frame)
            
            num_hands = len(all_landmarks)
            if num_hands > 0:
                detected_count += 1
                for landmarks in all_landmarks:
                    frame = draw_landmarks(frame, landmarks)
                status = f"{num_hands} hand(s) detected | {num_hands * 63} features"
            else:
                status = "No hand detected"
            
            detection_rate = (detected_count / total_count * 100) if total_count > 0 else 0
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detection: {detection_rate:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("ASL Translator - Landmarks", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"Detection rate: {detected_count}/{total_count} ({detection_rate:.1f}%)")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
