from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


# =========================================================
# 1. 경로 설정
# =========================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
TRAIN_IMAGES = DATASET_ROOT / "train" / "images"


# =========================================================
# 2. 클래스 목록 확인
# =========================================================

classes = sorted(
    item.name
    for item in TRAIN_IMAGES.iterdir()
    if item.is_dir()
)

print("Classes:")
for class_name in classes:
    print(f"- {class_name}")


# =========================================================
# 3. 전체 이미지 크기 / 모드 분포 확인
# =========================================================

size_counts = {}
mode_counts = {}

for class_name in classes:
    class_dir = TRAIN_IMAGES / class_name

    image_files = [
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    for image_path in image_files:
        with Image.open(image_path) as image:
            size_counts[image.size] = size_counts.get(image.size, 0) + 1
            mode_counts[image.mode] = mode_counts.get(image.mode, 0) + 1


print("\nImage size distribution:")
for size, count in size_counts.items():
    print(f"{size}: {count}")


print("\nImage mode distribution:")
for mode, count in mode_counts.items():
    print(f"{mode}: {count}")


# =========================================================
# 4. 클래스별 샘플 이미지 시각화
# =========================================================

samples_per_class = 3

fig, axes = plt.subplots(
    len(classes),
    samples_per_class,
    figsize=(9, 12)
)

for row, class_name in enumerate(classes):
    class_dir = TRAIN_IMAGES / class_name

    image_files = sorted(
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    )

    for col in range(samples_per_class):
        image_path = image_files[col]

        with Image.open(image_path) as image:
            axes[row, col].imshow(image)

        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_title(class_name)


plt.suptitle("Sample Images by Class")
plt.tight_layout()
plt.show()


# =========================================================
# 5. 클래스별 평균 밝기 계산
# =========================================================

brightness_by_class = {}
brightness_values_by_class = {}

for class_name in classes:
    class_dir = TRAIN_IMAGES / class_name

    image_files = [
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    brightness_values = []

    for image_path in image_files:
        with Image.open(image_path) as image:
            image_array = np.array(image)

            mean_brightness = image_array.mean()
            brightness_values.append(mean_brightness)

    brightness_by_class[class_name] = np.mean(brightness_values)
    brightness_values_by_class[class_name] = brightness_values


print("\nAverage brightness by class:")

for class_name, brightness in brightness_by_class.items():
    print(f"{class_name}: {brightness:.2f}")


# =========================================================
# 6. 클래스별 밝기 분포 히스토그램
# =========================================================

plt.figure(figsize=(10, 6))

for class_name, values in brightness_values_by_class.items():
    plt.hist(
        values,
        bins=20,
        alpha=0.5,
        label=class_name
    )

plt.xlabel("Average Brightness")
plt.ylabel("Number of Images")
plt.title("Brightness Distribution by Class")
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# 7. 클래스별 Edge 비교
# =========================================================

fig, axes = plt.subplots(
    len(classes),
    2,
    figsize=(8, 16)
)

for row, class_name in enumerate(classes):
    class_dir = TRAIN_IMAGES / class_name

    image_files = sorted(
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    )

    image_path = image_files[0]

    with Image.open(image_path) as image:
        gray_image = image.convert("L")
        edge_image = gray_image.filter(ImageFilter.FIND_EDGES)

        axes[row, 0].imshow(gray_image, cmap="gray")
        axes[row, 1].imshow(edge_image, cmap="gray")

    axes[row, 0].axis("off")
    axes[row, 1].axis("off")

    axes[row, 0].set_title(f"{class_name} - Original")
    axes[row, 1].set_title(f"{class_name} - Edge")


plt.suptitle("Edge Comparison by Class", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


# =========================================================
# 8. 데이터 품질 검사
# =========================================================

total_images = 0
valid_images = 0
invalid_images = []

for class_name in classes:
    class_dir = TRAIN_IMAGES / class_name

    image_files = [
        file
        for file in class_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    for image_path in image_files:
        total_images += 1

        try:
            with Image.open(image_path) as image:
                image.verify()

            valid_images += 1

        except Exception as error:
            invalid_images.append(
                (class_name, image_path.name, str(error))
            )


print("\n========================================")
print("Image Quality Check")
print("========================================")

print(f"Total images: {total_images}")
print(f"Valid images: {valid_images}")
print(f"Invalid images: {len(invalid_images)}")


if invalid_images:
    print("\nInvalid image list:")

    for class_name, filename, error in invalid_images:
        print(f"[{class_name}] {filename}")
        print(f"Error: {error}")
        print()

else:
    print("\nAll images are valid.")