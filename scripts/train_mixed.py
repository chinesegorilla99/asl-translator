import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json


def load_kaggle_data(project_root):
    data_dir = project_root / "data" / "processed_normalized"
    
    if not (data_dir / "features.npy").exists():
        print("Kaggle normalized data not found!")
        return None, None
    
    features = np.load(data_dir / "features.npy")
    labels = np.load(data_dir / "labels.npy")
    
    print(f"Kaggle data: {len(features)} samples")
    return features, labels


def load_personal_data(project_root):
    data_dir = project_root / "data" / "personal"
    
    if not (data_dir / "features_normalized.npy").exists():
        print("Personal data not found - run collect_personal_data.py first")
        return None, None
    
    features = np.load(data_dir / "features_normalized.npy")
    labels = np.load(data_dir / "labels.npy")
    
    print(f"Personal data: {len(features)} samples")
    return features, labels


def combine_datasets(kaggle_features, kaggle_labels, personal_features, personal_labels, personal_weight=2):
    combined_features = []
    combined_labels = []
    
    if kaggle_features is not None:
        combined_features.append(kaggle_features)
        combined_labels.append(kaggle_labels)
    
    if personal_features is not None:
        for _ in range(personal_weight):
            combined_features.append(personal_features)
            combined_labels.append(personal_labels)
    
    if not combined_features:
        return None, None
    
    features = np.vstack(combined_features)
    labels = np.concatenate(combined_labels)
    
    print(f"Combined data: {len(features)} samples")
    return features, labels


def train_model(features, labels, output_dir):
    print(f"\nDataset: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Classes: {len(set(labels))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
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
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "num_classes": len(unique_labels),
        "num_features": int(features.shape[1]),
        "label_to_idx": label_to_idx,
        "normalized": True,
        "normalization_method": "wrist_centered_palm_scaled",
        "training_data": "kaggle+personal"
    }
    
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"\nModel saved to: {output_dir / 'classifier.joblib'}")
    
    print("\n" + "=" * 50)
    print("Classification Report (Test Set)")
    print("=" * 50)
    print(classification_report(y_test, test_pred))
    
    return model, test_acc


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "models" / "saved"
    
    print("=" * 60)
    print("MIXED DATA TRAINING")
    print("=" * 60)
    
    kaggle_features, kaggle_labels = load_kaggle_data(project_root)
    personal_features, personal_labels = load_personal_data(project_root)
    
    if kaggle_features is None and personal_features is None:
        print("No training data available!")
        return
    
    print("\n--- Training on Kaggle data only ---")
    if kaggle_features is not None:
        train_model(kaggle_features, kaggle_labels, output_dir)
    
    if personal_features is not None:
        print("\n\n--- Training on Combined data (personal weighted 2x) ---")
        combined_features, combined_labels = combine_datasets(
            kaggle_features, kaggle_labels,
            personal_features, personal_labels,
            personal_weight=2
        )
        train_model(combined_features, combined_labels, output_dir)


if __name__ == "__main__":
    main()
