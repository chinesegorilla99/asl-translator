import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.preprocessing.data_splitter import load_train_test_split
from backend.training.trainer import get_models_path


def load_model():
    model_dir = get_models_path()
    model = joblib.load(model_dir / "classifier.joblib")
    
    scaler = None
    scaler_path = model_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    
    with open(model_dir / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    return model, scaler, metadata


def evaluate_model():
    model, scaler, model_metadata = load_model()
    X_train, X_test, y_train, y_test, split_metadata = load_train_test_split()
    
    if scaler is not None:
        X_test = scaler.transform(X_test)
        X_train = scaler.transform(X_train)
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    
    idx_to_label = {int(k): v for k, v in model_metadata["idx_to_label"].items()}
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    
    cm = confusion_matrix(y_test, y_pred)
    
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confused_pairs.append({
                    "true": labels[i],
                    "predicted": labels[j],
                    "count": int(cm[i][j])
                })
    
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    
    evaluation = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "per_class_metrics": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"])
            }
            for label in labels
        },
        "top_confusions": confused_pairs[:20],
        "confusion_matrix": cm.tolist()
    }
    
    output_dir = get_models_path()
    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    
    return evaluation


def print_evaluation(evaluation):
    print(f"Train Accuracy: {evaluation['train_accuracy']:.4f}")
    print(f"Test Accuracy:  {evaluation['test_accuracy']:.4f}")
    print(f"\nTest Samples: {evaluation['test_samples']}")
    
    print("\nTop Confusion Pairs:")
    for pair in evaluation["top_confusions"][:10]:
        print(f"  {pair['true']} -> {pair['predicted']}: {pair['count']}")
    
    print("\nWorst Performing Classes (by F1):")
    sorted_classes = sorted(
        evaluation["per_class_metrics"].items(),
        key=lambda x: x[1]["f1"]
    )
    for label, metrics in sorted_classes[:5]:
        print(f"  {label}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")


if __name__ == "__main__":
    print("Evaluating model...\n")
    evaluation = evaluate_model()
    print_evaluation(evaluation)
    print(f"\nEvaluation saved to: {get_models_path() / 'evaluation.json'}")