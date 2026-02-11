import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import joblib
import json
from datetime import datetime
from collections import defaultdict

WINDOW_NAME = "ASL Confusion Analyzer"
VALID_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


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


class ConfusionAnalyzer:
    def __init__(self):
        project_root = Path(__file__).parent.parent.parent
        
        model_path = project_root / "models" / "saved" / "classifier.joblib"
        metadata_path = project_root / "models" / "saved" / "model_metadata.json"
        hand_model_path = project_root / "models" / "mediapipe" / "hand_landmarker.task"
        
        self.model = joblib.load(model_path)
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        self.idx_to_label = {v: k for k, v in metadata["label_to_idx"].items()}
        self.normalized = metadata.get("normalized", False)
        
        base_options = python.BaseOptions(model_asset_path=str(hand_model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.predictions = []
        self.actual_labels = []
        self.current_prediction = None
        self.current_confidence = 0.0
        self.recording = False
        self.current_actual = None
        
    def detect_and_predict(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        
        if not result.hand_landmarks:
            self.current_prediction = None
            self.current_confidence = 0.0
            return None
        
        landmarks = result.hand_landmarks[0]
        coords = np.array([c for lm in landmarks for c in [lm.x, lm.y, lm.z]], dtype=np.float32)
        
        if self.normalized:
            coords = normalize_landmarks(coords)
        
        features = coords.reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_label = self.idx_to_label[predicted_idx].upper()
        
        self.current_prediction = predicted_label
        self.current_confidence = confidence
        
        return landmarks
    
    def record_sample(self, actual_label):
        if self.current_prediction is not None:
            self.predictions.append(self.current_prediction)
            self.actual_labels.append(actual_label.upper())
            print(f"Recorded: Actual={actual_label.upper()}, Predicted={self.current_prediction}, Conf={self.current_confidence:.2f}")
            return True
        return False
    
    def get_confusion_matrix(self):
        if not self.predictions:
            return {}
        
        confusion = defaultdict(lambda: defaultdict(int))
        for actual, predicted in zip(self.actual_labels, self.predictions):
            confusion[actual][predicted] += 1
        
        return dict(confusion)
    
    def print_results(self):
        if not self.predictions:
            print("No samples recorded!")
            return
        
        correct = sum(1 for a, p in zip(self.actual_labels, self.predictions) if a == p)
        total = len(self.predictions)
        accuracy = correct / total * 100
        
        print("\n" + "=" * 60)
        print("CONFUSION ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Total samples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.1f}%")
        
        confusion = self.get_confusion_matrix()
        
        print("\n--- Confusion Details ---")
        for actual in sorted(confusion.keys()):
            predictions = confusion[actual]
            total_for_actual = sum(predictions.values())
            correct_for_actual = predictions.get(actual, 0)
            
            print(f"\n{actual}: {correct_for_actual}/{total_for_actual} correct")
            
            for predicted, count in sorted(predictions.items()):
                if predicted != actual:
                    print(f"  -> Confused with {predicted}: {count} times")
        
        output_dir = Path(__file__).parent.parent.parent / "data" / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"confusion_{timestamp}.json"
        
        results = {
            "timestamp": timestamp,
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "predictions": self.predictions,
            "actual_labels": self.actual_labels,
            "confusion_matrix": {k: dict(v) for k, v in confusion.items()}
        }
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def close(self):
        self.detector.close()


def draw_ui(frame, analyzer, current_actual):
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    
    if analyzer.current_prediction:
        pred_text = f"Predicted: {analyzer.current_prediction}"
        conf_text = f"Confidence: {analyzer.current_confidence*100:.0f}%"
        
        color = (0, 255, 0) if analyzer.current_confidence >= 0.7 else (0, 255, 255)
        cv2.putText(frame, pred_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, conf_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, "No hand detected", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    if current_actual:
        cv2.putText(frame, f"Recording: {current_actual}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    samples = len(analyzer.predictions)
    cv2.putText(frame, f"Samples: {samples}", (w - 150, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 80), (w, h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0)
    
    cv2.putText(frame, "Type A-Z or 0-9 to record actual label | SPACE to save current | R for results | Q to quit", 
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Hold a sign, then press the letter/number you're actually signing", 
                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return frame


def main():
    print("=" * 60)
    print("ASL CONFUSION ANALYZER")
    print("=" * 60)
    print("""
Instructions:
1. Show an ASL sign to the camera
2. Press the key for what you're ACTUALLY signing (A-Z, 0-9)
3. This records the predicted vs actual label
4. Repeat for multiple signs
5. Press 'R' to see results
6. Press 'Q' to quit
""")
    
    analyzer = ConfusionAnalyzer()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)
    
    current_actual = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks = analyzer.detect_and_predict(frame)
            
            frame = draw_ui(frame, analyzer, current_actual)
            cv2.imshow(WINDOW_NAME, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                analyzer.print_results()
            elif chr(key).upper() in VALID_LABELS:
                actual = chr(key).upper()
                if analyzer.record_sample(actual):
                    current_actual = actual
                else:
                    print("No hand detected - cannot record")
            elif key == ord(' '):
                if current_actual and analyzer.current_prediction:
                    analyzer.record_sample(current_actual)
    
    finally:
        analyzer.print_results()
        analyzer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
