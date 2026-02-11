import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed_normalized"
    output_dir = project_root / "models" / "saved"
    
    features = np.load(data_dir / "features.npy")
    labels = np.load(data_dir / "labels.npy")
    
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Classes: {len(set(labels))}")
    print(f"Normalization: {metadata.get('normalization_method', 'none')}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    joblib.dump(model, output_dir / "classifier.joblib")
    
    model_metadata = {
        "model_type": "random_forest",
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "num_classes": len(unique_labels),
        "num_features": features.shape[1],
        "label_to_idx": label_to_idx,
        "normalized": True,
        "normalization_method": "wrist_centered_palm_scaled"
    }
    
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"\nModel saved to: {output_dir / 'classifier.joblib'}")
    print(f"Metadata saved to: {output_dir / 'model_metadata.json'}")
    
    print("\n" + "=" * 50)
    print("Classification Report (Test Set)")
    print("=" * 50)
    print(classification_report(y_test, test_pred))


if __name__ == "__main__":
    main()
