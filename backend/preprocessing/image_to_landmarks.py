import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.preprocessing.image_loader import get_all_image_paths, ASL_CLASSES
from backend.hand_tracking.landmark_extractor import create_hand_detector, extract_landmarks_from_image


def get_processed_data_path() -> Path:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root / "data" / "processed" / "landmarks"


def extract_and_save_landmarks(output_dir: Path = None):
    if output_dir is None:
        output_dir = get_processed_data_path()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = get_all_image_paths()
    detector = create_hand_detector()
    
    features = []
    labels = []
    skipped = []
    
    for label, image_path in tqdm(all_images, desc="Extracting landmarks"):
        landmarks = extract_landmarks_from_image(image_path, detector)
        
        if landmarks is not None:
            features.append(landmarks)
            labels.append(label)
        else:
            skipped.append(str(image_path))
    
    detector.close()
    
    features_array = np.array(features, dtype=np.float32)
    labels_array = np.array(labels, dtype=str)
    
    np.save(output_dir / "features.npy", features_array)
    np.save(output_dir / "labels.npy", labels_array)
    
    label_to_idx = {label: idx for idx, label in enumerate(ASL_CLASSES)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    metadata = {
        "num_samples": len(features),
        "num_features": 63,
        "num_classes": len(ASL_CLASSES),
        "classes": ASL_CLASSES,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "skipped_count": len(skipped),
        "numpy_version": np.__version__
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    if skipped:
        with open(output_dir / "skipped.txt", "w") as f:
            f.write("\n".join(skipped))
    
    return features_array, labels_array, metadata


def load_landmarks(data_dir: Path = None):
    if data_dir is None:
        data_dir = get_processed_data_path()
    
    features = np.load(data_dir / "features.npy")
    labels = np.load(data_dir / "labels.npy", allow_pickle=True)
    
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    return features, labels, metadata


if __name__ == "__main__":
    print("Extracting landmarks from dataset...")
    features, labels, metadata = extract_and_save_landmarks()
    
    print(f"\nSaved to: {get_processed_data_path()}")
    print(f"Samples: {metadata['num_samples']}")
    print(f"Features per sample: {metadata['num_features']}")
    print(f"Classes: {metadata['num_classes']}")
    print(f"Skipped: {metadata['skipped_count']}")