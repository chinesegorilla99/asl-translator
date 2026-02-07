import os
from pathlib import Path


ASL_CLASSES = list("abcdefghijklmnopqrstuvwxyz") + list("0123456789")


def get_dataset_path() -> Path:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    dataset_path = project_root / "data" / "raw" / "kaggle_asl_dataset"
    return dataset_path


def get_class_directories(dataset_path: Path = None) -> dict[str, Path]:
    if dataset_path is None:
        dataset_path = get_dataset_path()
    
    class_dirs = {}
    for label in ASL_CLASSES:
        class_dir = dataset_path / label
        if class_dir.exists() and class_dir.is_dir():
            class_dirs[label] = class_dir
    
    return class_dirs


def get_image_paths_for_class(class_dir: Path) -> list[Path]:
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = []
    
    for file_path in class_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_paths.append(file_path)
    
    return sorted(image_paths)


def get_all_image_paths(dataset_path: Path = None) -> list[tuple[str, Path]]:
    class_dirs = get_class_directories(dataset_path)
    all_images = []
    
    for label, class_dir in class_dirs.items():
        image_paths = get_image_paths_for_class(class_dir)
        for path in image_paths:
            all_images.append((label, path))
    
    return all_images


def get_dataset_summary(dataset_path: Path = None) -> dict:
    class_dirs = get_class_directories(dataset_path)
    class_counts = {}
    
    for label, class_dir in class_dirs.items():
        image_paths = get_image_paths_for_class(class_dir)
        class_counts[label] = len(image_paths)
    
    return {
        "class_counts": class_counts,
        "total_images": sum(class_counts.values()),
        "num_classes": len(class_counts)
    }


if __name__ == "__main__":
    print("Dataset path:", get_dataset_path())
    print()
    
    summary = get_dataset_summary()
    print(f"Total classes: {summary['num_classes']}")
    print(f"Total images: {summary['total_images']}")
    print()
    
    print("Images per class:")
    for label, count in sorted(summary["class_counts"].items()):
        print(f"  {label}: {count}")
