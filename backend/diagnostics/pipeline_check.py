import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import json


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


def compare_pipelines():
    print("=" * 60)
    print("LANDMARK PIPELINE CONSISTENCY CHECK")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent.parent
    
    features_path = project_root / "data" / "processed" / "features.npy"
    metadata_path = project_root / "data" / "processed" / "metadata.json"
    
    if not features_path.exists():
        print("ERROR: Training features not found")
        return
    
    training_features = np.load(features_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"\n[Training Data]")
    print(f"  Shape: {training_features.shape}")
    print(f"  Dtype: {training_features.dtype}")
    print(f"  Landmarks per sample: {training_features.shape[1] // 3}")
    print(f"  Coords per landmark: 3 (x, y, z)")
    print(f"  Total features: {training_features.shape[1]}")
    
    sample = training_features[0].reshape(21, 3)
    print(f"\n[Training Sample Statistics]")
    print(f"  X range: [{sample[:, 0].min():.4f}, {sample[:, 0].max():.4f}]")
    print(f"  Y range: [{sample[:, 1].min():.4f}, {sample[:, 1].max():.4f}]")
    print(f"  Z range: [{sample[:, 2].min():.4f}, {sample[:, 2].max():.4f}]")
    print(f"  Wrist (landmark 0): [{sample[0, 0]:.4f}, {sample[0, 1]:.4f}, {sample[0, 2]:.4f}]")
    
    print("\n" + "=" * 60)
    print("ISSUE ANALYSIS")
    print("=" * 60)
    
    print("""
IDENTIFIED ISSUES:

1. NO NORMALIZATION
   - Training data uses raw MediaPipe coordinates (0-1 range)
   - Real-time uses same raw coordinates
   - BUT: Hand position, size, and distance vary in real-time
   - Result: Same sign at different positions = different features

2. HANDEDNESS NOT HANDLED
   - Training images may show right hands
   - Your webcam may detect left hand (or mirrored right)
   - MediaPipe reports handedness but we're not using it
   - L and Q confusion could be due to hand orientation

3. IMAGE DOMAIN MISMATCH
   - Training: Kaggle static images (white background, centered)
   - Real-time: Webcam (varied background, lighting, position)
   - Training used preprocessing variants (padding, brightness)
   - Real-time does NOT use these variants

4. CAMERA MIRRORING
   - Webcam is flipped horizontally for natural viewing
   - This effectively mirrors the hand
   - A right hand appears as a left hand
   - Signs that differ by orientation (L/Q, etc.) get confused
""")
    
    return training_features


def test_realtime_extraction():
    print("\n" + "=" * 60)
    print("TESTING REAL-TIME EXTRACTION")
    print("=" * 60)
    
    model_path = Path(__file__).parent.parent.parent / "models" / "mediapipe" / "hand_landmarker.task"
    
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Show your hand to the camera. Press 'c' to capture, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        cv2.putText(display, "Press 'c' to capture landmarks, 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture Test", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)
            
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name if result.handedness else "Unknown"
                
                coords = []
                for lm in landmarks:
                    coords.extend([lm.x, lm.y, lm.z])
                coords = np.array(coords, dtype=np.float32)
                
                sample = coords.reshape(21, 3)
                print(f"\n[Real-time Capture]")
                print(f"  Handedness: {handedness}")
                print(f"  Features: {len(coords)}")
                print(f"  X range: [{sample[:, 0].min():.4f}, {sample[:, 0].max():.4f}]")
                print(f"  Y range: [{sample[:, 1].min():.4f}, {sample[:, 1].max():.4f}]")
                print(f"  Z range: [{sample[:, 2].min():.4f}, {sample[:, 2].max():.4f}]")
                print(f"  Wrist: [{sample[0, 0]:.4f}, {sample[0, 1]:.4f}, {sample[0, 2]:.4f}]")
                
                normalized = normalize_landmarks(coords)
                norm_sample = normalized.reshape(21, 3)
                print(f"\n[After Normalization]")
                print(f"  X range: [{norm_sample[:, 0].min():.4f}, {norm_sample[:, 0].max():.4f}]")
                print(f"  Y range: [{norm_sample[:, 1].min():.4f}, {norm_sample[:, 1].max():.4f}]")
                print(f"  Z range: [{norm_sample[:, 2].min():.4f}, {norm_sample[:, 2].max():.4f}]")
                print(f"  Wrist: [{norm_sample[0, 0]:.4f}, {norm_sample[0, 1]:.4f}, {norm_sample[0, 2]:.4f}]")
            else:
                print("\nNo hand detected!")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    compare_pipelines()
    test_realtime_extraction()
