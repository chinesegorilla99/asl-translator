import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

CONFUSING_LETTERS = ['a', 's', 'm', 'n', 't']
CONFUSING_WEIGHT_BOOST = 3.0


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


def augment_landmarks(landmarks, num_augments=5):
    augmented = [landmarks]
    coords = landmarks.reshape(21, 3)
    
    for _ in range(num_augments):
        aug_coords = coords.copy()
        
        angle = np.random.uniform(-20, 20) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        aug_coords[:, :2] = aug_coords[:, :2] @ rotation_matrix.T
        
        tilt_angle = np.random.uniform(-10, 10) * np.pi / 180
        cos_t, sin_t = np.cos(tilt_angle), np.sin(tilt_angle)
        tilt_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        aug_coords[:, [0, 2]] = aug_coords[:, [0, 2]] @ tilt_matrix.T
        
        scale = np.random.uniform(0.85, 1.15)
        aug_coords *= scale
        
        offset = np.random.uniform(-0.03, 0.03, size=3)
        aug_coords += offset
        
        brightness_shift = np.random.uniform(-0.05, 0.05)
        aug_coords[:, 2] += brightness_shift
        
        noise = np.random.normal(0, 0.005, aug_coords.shape)
        aug_coords += noise
        
        augmented.append(aug_coords.flatten().astype(np.float32))
    
    return augmented


def apply_augmentation(features, labels, augment_all=True, extra_for_confusing=True):
    aug_features = []
    aug_labels = []
    
    for feat, label in zip(features, labels):
        aug_features.append(feat)
        aug_labels.append(label)
        
        if augment_all:
            num_aug = 5 if (extra_for_confusing and label.lower() in CONFUSING_LETTERS) else 2
            for aug in augment_landmarks(feat, num_augments=num_aug):
                if not np.array_equal(aug, feat):
                    aug_features.append(aug)
                    aug_labels.append(label)
        elif extra_for_confusing and label.lower() in CONFUSING_LETTERS:
            for aug in augment_landmarks(feat, num_augments=5):
                if not np.array_equal(aug, feat):
                    aug_features.append(aug)
                    aug_labels.append(label)
    
    return np.array(aug_features), np.array(aug_labels)


def compute_class_weights(labels):
    unique_labels = np.unique(labels)
    n_samples = len(labels)
    n_classes = len(unique_labels)
    
    weights = {}
    for label in unique_labels:
        count = np.sum(labels == label)
        weights[label] = n_samples / (n_classes * count)
        
        if label.lower() in CONFUSING_LETTERS:
            weights[label] *= CONFUSING_WEIGHT_BOOST
    
    return weights


def train_model(features, labels, output_dir, use_augmentation=True, augment_all=True, max_iterations=50):
    if use_augmentation:
        print("Applying data augmentation (rotation, tilt, scale, noise, brightness)...")
        features, labels = apply_augmentation(features, labels, augment_all=augment_all)
        print(f"After augmentation: {len(features)} samples")
    
    print(f"\nDataset: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Classes: {len(set(labels))}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    class_weights = compute_class_weights(y_train)
    print(f"\nBoosted weights for confusing letters: {CONFUSING_LETTERS}")
    
    print("\nTraining with early stopping...")
    best_model = None
    best_val_acc = 0.0
    patience = 5
    no_improve_count = 0
    
    for n_trees in range(50, 50 + max_iterations * 50, 50):
        model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight=class_weights
        )
        
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val))
        
        print(f"  Trees: {n_trees}, Val Accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"  Early stopping at {n_trees} trees")
            break
    
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    joblib.dump(best_model, output_dir / "classifier.joblib")
    
    model_metadata = {
        "model_type": "random_forest",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "val_accuracy": float(best_val_acc),
        "num_classes": len(unique_labels),
        "num_features": int(features.shape[1]),
        "label_to_idx": label_to_idx,
        "normalized": True,
        "normalization_method": "wrist_centered_palm_scaled",
        "training_data": "kaggle+personal",
        "augmented": use_augmentation,
        "confusing_letters_boosted": CONFUSING_LETTERS,
        "early_stopping": True
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
        train_model(kaggle_features, kaggle_labels, output_dir, use_augmentation=True)
    
    if personal_features is not None:
        print("\n\n--- Training on Combined data (personal weighted 2x) ---")
        combined_features, combined_labels = combine_datasets(
            kaggle_features, kaggle_labels,
            personal_features, personal_labels,
            personal_weight=2
        )
        train_model(combined_features, combined_labels, output_dir, use_augmentation=True)


if __name__ == "__main__":
    main()
