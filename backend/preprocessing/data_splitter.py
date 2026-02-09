import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.preprocessing.image_to_landmarks import load_landmarks, get_processed_data_path
from backend.preprocessing.image_loader import ASL_CLASSES


def get_splits_path() -> Path:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root / "data" / "processed" / "splits"


def create_train_test_split(test_size: float = 0.2, random_state: int = 42):
    features, labels, metadata = load_landmarks()
    
    label_to_idx = {label: idx for idx, label in enumerate(ASL_CLASSES)}
    labels_encoded = np.array([label_to_idx[label] for label in labels], dtype=np.int32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_encoded
    )
    
    output_dir = get_splits_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)
    
    split_metadata = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "num_features": X_train.shape[1],
        "num_classes": len(ASL_CLASSES),
        "label_to_idx": label_to_idx,
        "idx_to_label": {idx: label for label, idx in label_to_idx.items()},
        "numpy_version": np.__version__
    }
    
    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2)
    
    return X_train, X_test, y_train, y_test, split_metadata


def load_train_test_split(data_dir: Path = None):
    if data_dir is None:
        data_dir = get_splits_path()
    
    X_train = np.load(data_dir / "X_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    with open(data_dir / "split_metadata.json", "r") as f:
        metadata = json.load(f)
    
    return X_train, X_test, y_train, y_test, metadata


if __name__ == "__main__":
    print("Creating train/test split...")
    X_train, X_test, y_train, y_test, metadata = create_train_test_split()
    
    print(f"\nSaved to: {get_splits_path()}")
    print(f"Train samples: {metadata['train_samples']}")
    print(f"Test samples: {metadata['test_samples']}")
    print(f"Features: {metadata['num_features']}")
    print(f"Classes: {metadata['num_classes']}")