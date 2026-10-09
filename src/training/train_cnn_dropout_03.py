from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# =========================================================
# 1. 경로 설정
# =========================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "NEU-DET"

TRAIN_DIR = DATASET_ROOT / "train" / "images"
VAL_DIR = DATASET_ROOT / "validation" / "images"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODEL_DIR / "cnn_dropout_03.keras"


# =========================================================
# 2. 기본 설정
# =========================================================

IMAGE_SIZE = (200, 200)
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 6
EPOCHS = 10
DROPOUT_RATE = 0.3


# =========================================================
# 3. 저장 폴더 생성
# =========================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 4. Random Seed 고정
# =========================================================

tf.keras.utils.set_random_seed(SEED)


# =========================================================
# 5. Train 데이터 로더
# =========================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)


# =========================================================
# 6. Validation 데이터 로더
# =========================================================

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)


# =========================================================
# 7. 클래스 확인
# =========================================================

class_names = train_dataset.class_names

print("\n========================================")
print("Dataset Information")
print("========================================")

print("Classes:", class_names)
print("Number of classes:", len(class_names))


# =========================================================
# 8. CNN + Dropout(0.3) 모델 생성
# =========================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(200, 200, 3)),

    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(DROPOUT_RATE),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# =========================================================
# 9. 모델 구조 확인
# =========================================================

print("\n========================================")
print("CNN Dropout 0.3 Model")
print("========================================")

model.summary()


# =========================================================
# 10. 모델 Compile
# =========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 11. 모델 학습
# =========================================================

print("\n========================================")
print("Training Start")
print("========================================")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)


# =========================================================
# 12. 학습 결과 확인
# =========================================================

train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]


print("\n========================================")
print("Training Complete")
print("========================================")

print(f"Dropout Rate         : {DROPOUT_RATE}")
print(f"Final Train Accuracy : {train_accuracy[-1]:.4f}")
print(f"Final Val Accuracy   : {val_accuracy[-1]:.4f}")
print(f"Final Train Loss     : {train_loss[-1]:.4f}")
print(f"Final Val Loss       : {val_loss[-1]:.4f}")


# =========================================================
# 13. 모델 저장
# =========================================================

model.save(MODEL_PATH)

print("\n========================================")
print("Model Save Complete")
print("========================================")

print(f"Saved model: {MODEL_PATH}")


# =========================================================
# 14. Accuracy 그래프
# =========================================================

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_accuracy,
    label="Train Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Dropout 0.3 - Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

accuracy_graph_path = RESULTS_DIR / "cnn_dropout_03_accuracy.png"
plt.savefig(accuracy_graph_path)

plt.show()


# =========================================================
# 15. Loss 그래프
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_loss,
    label="Train Loss"
)

plt.plot(
    epochs_range,
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Dropout 0.3 - Training and Validation Loss")
plt.legend()
plt.tight_layout()

loss_graph_path = RESULTS_DIR / "cnn_dropout_03_loss.png"
plt.savefig(loss_graph_path)

plt.show()


# =========================================================
# 16. Validation 데이터 예측
# =========================================================

true_labels = []
predicted_labels = []

for images, labels in val_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted_classes)


true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# =========================================================
# 17. Confusion Matrix 생성
# =========================================================

confusion_matrix = tf.math.confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes=NUM_CLASSES
).numpy()


print("\n========================================")
print("Confusion Matrix")
print("========================================")

print(confusion_matrix)


# =========================================================
# 18. Confusion Matrix 시각화
# =========================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    confusion_matrix,
    interpolation="nearest"
)

plt.title("CNN Dropout 0.3 - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(NUM_CLASSES),
    class_names,
    rotation=45
)

plt.yticks(
    range(NUM_CLASSES),
    class_names
)

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            confusion_matrix[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

confusion_matrix_path = (
    RESULTS_DIR / "cnn_dropout_03_confusion_matrix.png"
)

plt.savefig(confusion_matrix_path)

plt.show()


# =========================================================
# 19. 클래스별 Accuracy 계산
# =========================================================

print("\n========================================")
print("Accuracy by Class")
print("========================================")

for index, class_name in enumerate(class_names):

    class_total = confusion_matrix[index].sum()
    class_correct = confusion_matrix[index, index]

    class_accuracy = (
        class_correct / class_total
        if class_total > 0
        else 0
    )

    print(
        f"{class_name}: "
        f"{class_correct}/{class_total} "
        f"({class_accuracy:.4f})"
    )


# =========================================================
# 20. Confusion Matrix 기준 전체 Accuracy
# =========================================================

total_correct = np.trace(confusion_matrix)
total_samples = confusion_matrix.sum()

cm_accuracy = total_correct / total_samples

print("\n========================================")
print("Confusion Matrix Accuracy")
print("========================================")

print(f"Correct : {total_correct}/{total_samples}")
print(f"Accuracy: {cm_accuracy:.4f}")


# =========================================================
# 21. 세 모델 비교
# =========================================================

BASELINE_TRAIN_ACCURACY = 0.9542
BASELINE_VAL_ACCURACY = 0.8917
BASELINE_VAL_LOSS = 0.2497

DROPOUT_05_TRAIN_ACCURACY = 0.9049
DROPOUT_05_VAL_ACCURACY = 0.8528
DROPOUT_05_VAL_LOSS = 0.3221


print("\n========================================")
print("Baseline vs Dropout 0.5 vs Dropout 0.3")
print("========================================")

print(
    f"Baseline Train Accuracy    : "
    f"{BASELINE_TRAIN_ACCURACY:.4f}"
)

print(
    f"Dropout 0.5 Train Accuracy : "
    f"{DROPOUT_05_TRAIN_ACCURACY:.4f}"
)

print(
    f"Dropout 0.3 Train Accuracy : "
    f"{train_accuracy[-1]:.4f}"
)

print()

print(
    f"Baseline Val Accuracy      : "
    f"{BASELINE_VAL_ACCURACY:.4f}"
)

print(
    f"Dropout 0.5 Val Accuracy   : "
    f"{DROPOUT_05_VAL_ACCURACY:.4f}"
)

print(
    f"Dropout 0.3 Val Accuracy   : "
    f"{val_accuracy[-1]:.4f}"
)

print()

print(
    f"Baseline Val Loss          : "
    f"{BASELINE_VAL_LOSS:.4f}"
)

print(
    f"Dropout 0.5 Val Loss       : "
    f"{DROPOUT_05_VAL_LOSS:.4f}"
)

print(
    f"Dropout 0.3 Val Loss       : "
    f"{val_loss[-1]:.4f}"
)