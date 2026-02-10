import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.preprocessing.data_splitter import load_train_test_split


def get_models_path() -> Path:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root / "models" / "saved"


def train_random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
    return clf, None, train_acc, test_acc


def train_mlp(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    
    return clf, scaler, train_acc, test_acc


def train_and_compare():
    X_train, X_test, y_train, y_test, metadata = load_train_test_split()
    
    print("Training Random Forest...")
    rf_model, _, rf_train_acc, rf_test_acc = train_random_forest(X_train, y_train, X_test, y_test)
    print(f"  Train: {rf_train_acc:.4f}, Test: {rf_test_acc:.4f}")
    
    print("\nTraining MLP...")
    mlp_model, mlp_scaler, mlp_train_acc, mlp_test_acc = train_mlp(X_train, y_train, X_test, y_test)
    print(f"  Train: {mlp_train_acc:.4f}, Test: {mlp_test_acc:.4f}")
    
    if rf_test_acc >= mlp_test_acc:
        best_model = rf_model
        best_scaler = None
        best_name = "random_forest"
        best_acc = rf_test_acc
    else:
        best_model = mlp_model
        best_scaler = mlp_scaler
        best_name = "mlp"
        best_acc = mlp_test_acc
    
    print(f"\nBest model: {best_name} ({best_acc:.4f})")
    
    output_dir = get_models_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_model, output_dir / "classifier.joblib")
    if best_scaler is not None:
        joblib.dump(best_scaler, output_dir / "scaler.joblib")
    
    model_metadata = {
        "model_type": best_name,
        "train_accuracy": float(rf_train_acc if best_name == "random_forest" else mlp_train_acc),
        "test_accuracy": float(best_acc),
        "num_classes": metadata["num_classes"],
        "num_features": metadata["num_features"],
        "label_to_idx": metadata["label_to_idx"],
        "idx_to_label": metadata["idx_to_label"],
        "requires_scaling": best_scaler is not None,
        "comparison": {
            "random_forest": {"train": float(rf_train_acc), "test": float(rf_test_acc)},
            "mlp": {"train": float(mlp_train_acc), "test": float(mlp_test_acc)}
        }
    }
    
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2)
    
    return best_model, best_scaler, model_metadata


if __name__ == "__main__":
    model, scaler, metadata = train_and_compare()
    print(f"\nModel saved to: {get_models_path()}")