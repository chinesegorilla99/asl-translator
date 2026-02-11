import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import json
from datetime import datetime
import time
import argparse

WINDOW_NAME = "ASL Data Collection"
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]
DEFAULT_SAMPLES_PER_LABEL = 150
COUNTDOWN_SECONDS = 3
CAPTURE_DELAY_MS = 50


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
    
    palm_vector = index_mcp - pinky_mcp
    palm_vector_2d = palm_vector[:2]
    palm_norm = np.linalg.norm(palm_vector_2d)
    if palm_norm > 0.001:
        angle = np.arctan2(palm_vector_2d[1], palm_vector_2d[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normalized[:, :2] = normalized[:, :2] @ rotation.T
    
    return normalized.flatten().astype(np.float32)


class DataCollector:
    def __init__(self, append_mode=False):
        project_root = Path(__file__).parent.parent
        hand_model_path = project_root / "models" / "mediapipe" / "hand_landmarker.task"
        
        base_options = python.BaseOptions(model_asset_path=str(hand_model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.output_dir = project_root / "data" / "personal"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_features = []
        self.all_labels = []
        self.append_mode = append_mode
        
        if append_mode and (self.output_dir / "features_raw.npy").exists():
            existing_features = np.load(self.output_dir / "features_raw.npy")
            existing_labels = np.load(self.output_dir / "labels.npy")
            self.all_features = list(existing_features)
            self.all_labels = list(existing_labels)
            print(f"Loaded {len(self.all_features)} existing samples")
        
    def extract_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        
        if not result.hand_landmarks:
            return None
        
        landmarks = result.hand_landmarks[0]
        coords = np.array([c for lm in landmarks for c in [lm.x, lm.y, lm.z]], dtype=np.float32)
        
        return coords
    
    def collect_label(self, cap, label, samples_needed):
        collected = []
        
        print(f"\n{'='*50}")
        print(f"Collecting: {label}")
        print(f"{'='*50}")
        
        for countdown in range(COUNTDOWN_SECONDS, 0, -1):
            start_time = time.time()
            while time.time() - start_time < 1:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                display = frame.copy()
                h, w = display.shape[:2]
                
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
                display = cv2.addWeighted(overlay, 0.8, display, 0.2, 0)
                
                cv2.putText(display, f"Get ready to sign: {label}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(display, f"Starting in {countdown}...", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                cv2.imshow(WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return None
        
        last_capture = 0
        
        while len(collected) < samples_needed:
            ret, frame = cap.read()
            if not ret:
                continue
            
            current_time = cv2.getTickCount()
            
            landmarks = self.extract_landmarks(frame)
            
            display = frame.copy()
            h, w = display.shape[:2]
            
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
            display = cv2.addWeighted(overlay, 0.8, display, 0.2, 0)
            
            progress = len(collected) / samples_needed
            bar_width = int(400 * progress)
            cv2.rectangle(display, (20, 95), (420, 110), (50, 50, 50), -1)
            cv2.rectangle(display, (20, 95), (20 + bar_width, 110), (0, 255, 0), -1)
            
            cv2.putText(display, f"Signing: {label}", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(display, f"Collected: {len(collected)}/{samples_needed}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            if landmarks is None:
                cv2.putText(display, "NO HAND DETECTED - Show your hand!", (w//2 - 200, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                elapsed = (current_time - last_capture) / cv2.getTickFrequency() * 1000
                if elapsed >= CAPTURE_DELAY_MS:
                    collected.append(landmarks)
                    last_capture = current_time
            
            cv2.putText(display, "Press 'S' to skip | 'R' to redo | 'Q' to quit", (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow(WINDOW_NAME, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None
            elif key == ord('s'):
                print(f"Skipped {label}")
                return []
            elif key == ord('r'):
                print(f"Redoing {label}")
                return "redo"
        
        print(f"Collected {len(collected)} samples for {label}")
        return collected
    
    def save_data(self):
        if not self.all_features:
            print("No data to save!")
            return
        
        features = np.array(self.all_features, dtype=np.float32)
        labels = np.array(self.all_labels)
        
        normalized_features = np.zeros_like(features)
        for i in range(len(features)):
            normalized_features[i] = normalize_landmarks(features[i])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        np.save(self.output_dir / f"features_raw_{timestamp}.npy", features)
        np.save(self.output_dir / f"features_normalized_{timestamp}.npy", normalized_features)
        np.save(self.output_dir / f"labels_{timestamp}.npy", labels)
        
        np.save(self.output_dir / "features_raw.npy", features)
        np.save(self.output_dir / "features_normalized.npy", normalized_features)
        np.save(self.output_dir / "labels.npy", labels)
        
        metadata = {
            "timestamp": timestamp,
            "total_samples": len(features),
            "labels_collected": list(set(labels)),
            "samples_per_label": {label: int(np.sum(labels == label)) for label in set(labels)},
            "normalized": True,
            "normalization_method": "wrist_centered_palm_scaled"
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*50}")
        print("DATA SAVED")
        print(f"{'='*50}")
        print(f"Location: {self.output_dir}")
        print(f"Total samples: {len(features)}")
        print(f"Labels: {len(set(labels))}")
        print(f"Files:")
        print(f"  - features_raw.npy")
        print(f"  - features_normalized.npy")
        print(f"  - labels.npy")
        print(f"  - metadata.json")
    
    def close(self):
        self.detector.close()


def main():
    parser = argparse.ArgumentParser(description="Collect personal ASL training data")
    parser.add_argument("--letters", type=str, default=None,
                        help="Specific letters to collect (e.g., 'ASMNT' for confusing letters)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_LABEL,
                        help=f"Samples per letter (default: {DEFAULT_SAMPLES_PER_LABEL})")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing data instead of overwriting")
    args = parser.parse_args()
    
    if args.letters:
        target_labels = [c.upper() for c in args.letters]
    else:
        target_labels = LABELS
    
    samples_per_label = args.samples
    
    print("=" * 60)
    print("ASL PERSONAL DATA COLLECTION")
    print("=" * 60)
    print(f"""
This tool will collect {samples_per_label} samples for each letter/number.

{"Collecting specific letters: " + ", ".join(target_labels) if args.letters else "Collecting all letters and numbers"}

Instructions:
1. For each letter, you'll get a {COUNTDOWN_SECONDS}-second countdown
2. Hold the sign steady while samples are captured
3. Move your hand slightly for variation (position, angle)
4. Press 'S' to skip a letter
5. Press 'Q' to quit and save
6. Press 'R' to redo the current letter

{"Mode: APPEND to existing data" if args.append else "Mode: OVERWRITE existing data"}
""")
    
    input("Press Enter to start...")
    
    collector = DataCollector(append_mode=args.append)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)
    
    try:
        i = 0
        while i < len(target_labels):
            label = target_labels[i]
            samples = collector.collect_label(cap, label, samples_per_label)
            
            if samples is None:
                print("\nCollection cancelled by user")
                break
            
            if samples == "redo":
                continue
            
            for sample in samples:
                collector.all_features.append(sample)
                collector.all_labels.append(label.lower())
            
            i += 1
        
        collector.save_data()
        
    finally:
        collector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
