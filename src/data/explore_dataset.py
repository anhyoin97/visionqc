from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "NEU-DET"

TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
TRAIN_ANNOTATIONS = DATASET_ROOT / "train" / "annotations"

VAL_IMAGES = DATASET_ROOT / "validation" / "images"
VAL_ANNOTATIONS = DATASET_ROOT / "validation" / "annotations"


if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")

classes = sorted(
    item.name
    for item in TRAIN_IMAGES.iterdir()
    if item.is_dir()
)

print("Dataset root:", DATASET_ROOT)
print("Classes:", classes)
print("Class count:", len(classes))

for class_name in classes:
    class_dir = TRAIN_IMAGES / class_name

    image_files = list(class_dir.glob("*"))

    print(f"{class_name}: {len(image_files)}")