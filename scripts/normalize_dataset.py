import numpy as np
from pathlib import Path
import sys
import json
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.preprocessing.normalize import normalize_batch


def main():
    processed_dir = project_root / "data" / "processed" / "landmarks"
    
    features = np.load(processed_dir / "features.npy")
    labels = np.load(processed_dir / "labels.npy")
    
    print(f"Original features shape: {features.shape}")
    print(f"Original features dtype: {features.dtype}")
    
    sample = features[0].reshape(21, 3)
    print(f"\nBefore normalization (sample 0):")
    print(f"  Wrist: [{sample[0, 0]:.4f}, {sample[0, 1]:.4f}, {sample[0, 2]:.4f}]")
    print(f"  X range: [{sample[:, 0].min():.4f}, {sample[:, 0].max():.4f}]")
    
    print("\nNormalizing all features...")
    normalized_features = normalize_batch(features)
    
    norm_sample = normalized_features[0].reshape(21, 3)
    print(f"\nAfter normalization (sample 0):")
    print(f"  Wrist: [{norm_sample[0, 0]:.4f}, {norm_sample[0, 1]:.4f}, {norm_sample[0, 2]:.4f}]")
    print(f"  X range: [{norm_sample[:, 0].min():.4f}, {norm_sample[:, 0].max():.4f}]")
    
    output_dir = project_root / "data" / "processed_normalized"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "features.npy", normalized_features)
    np.save(output_dir / "labels.npy", labels)
    
    shutil.copy(processed_dir / "metadata.json", output_dir / "metadata.json")
    
    with open(output_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    metadata["normalized"] = True
    metadata["normalization_method"] = "wrist_centered_palm_scaled"
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nNormalized data saved to: {output_dir}")
    print(f"  features.npy: {normalized_features.shape}")
    print(f"  labels.npy: {labels.shape}")


if __name__ == "__main__":
    main()
