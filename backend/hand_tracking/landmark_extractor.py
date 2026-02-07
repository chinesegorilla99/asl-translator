import mediapipe as mp
import cv2
import numpy as np
from pathlib import Path


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def get_model_path() -> Path:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root / "models" / "mediapipe" / "hand_landmarker.task"


def create_hand_detector(max_hands: int = 1, min_detection_confidence: float = 0.1, min_tracking_confidence: float = 0.1):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(get_model_path())),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=max_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_tracking_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    return HandLandmarker.create_from_options(options)


def preprocess_image(image: np.ndarray) -> list[np.ndarray]:
    variants = []
    
    padded = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    variants.append(padded)
    
    bright = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
    variants.append(bright)
    
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (w * 2, h * 2))
    variants.append(scaled)
    
    padded_large = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    variants.append(padded_large)
    
    contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    variants.append(contrast)
    
    scaled_padded = cv2.copyMakeBorder(scaled, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    variants.append(scaled_padded)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    variants.append(equalized_bgr)
    
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    variants.append(sharpened)
    
    return variants


def extract_landmarks_from_image(image_path: Path, detector, try_variants: bool = True) -> list[float] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    results = detector.detect(mp_image)
    
    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    if try_variants:
        for variant in preprocess_image(image)[1:]:
            variant_rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=variant_rgb)
            results = detector.detect(mp_image)
            
            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                landmarks = []
                for landmark in hand_landmarks:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                return landmarks
    
    return None


def extract_landmarks_from_array(image_array: np.ndarray, detector) -> list[float] | None:
    if image_array is None:
        return None
    
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_array
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect(mp_image)
    
    if not results.hand_landmarks:
        return None
    
    hand_landmarks = results.hand_landmarks[0]
    landmarks = []
    
    for landmark in hand_landmarks:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return landmarks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from backend.preprocessing.image_loader import get_all_image_paths
    
    all_images = get_all_image_paths()
    print(f"Total images: {len(all_images)}")
    
    detector = create_hand_detector()
    
    success_count = 0
    fail_count = 0
    
    for i, (label, image_path) in enumerate(all_images[:100]):
        landmarks = extract_landmarks_from_image(image_path, detector)
        
        if landmarks:
            success_count += 1
            if i < 3:
                print(f"{label}: {len(landmarks)} values (21 landmarks x 3 coords)")
        else:
            fail_count += 1
    
    detector.close()
    
    print(f"\nResults (first 100 images):")
    print(f"  Success: {success_count}")
    print(f"  No hand detected: {fail_count}")