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

BEST_MODEL_PATH = (
    MODEL_DIR / "cnn_earlystop_best.keras"
)


# =========================================================
# 2. 기본 설정
# =========================================================

IMAGE_SIZE = (200, 200)
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 6

# EarlyStopping을 사용하므로 충분히 크게 설정
EPOCHS = 30


# =========================================================
# 3. 저장 폴더 생성
# =========================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 4. Random Seed 고정
# =========================================================

tf.keras.utils.set_random_seed(
    SEED
)


# =========================================================
# 5. Train 데이터 로더
# =========================================================

train_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )
)


# =========================================================
# 6. Validation 데이터 로더
# =========================================================

val_dataset = (
    tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
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
# 8. CNN Baseline 구조 생성
# =========================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(200, 200, 3)
    ),

    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

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

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# =========================================================
# 9. Compile
# =========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# 10. Callback 설정
# =========================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    # Validation Loss 기준
    monitor="val_loss",

    # 낮아지는 것이 개선
    mode="min",

    # 3 Epoch 동안 개선되지 않으면 종료
    patience=3,

    # 가장 좋은 Epoch의 Weight 복구
    restore_best_weights=True,

    verbose=1
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath=BEST_MODEL_PATH,

    # Validation Loss 기준
    monitor="val_loss",

    mode="min",

    # 최고 성능 모델만 저장
    save_best_only=True,

    verbose=1
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

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# =========================================================
# 12. 실제 학습 Epoch 확인
# =========================================================

actual_epochs = len(
    history.history["loss"]
)

print("\n========================================")
print("EarlyStopping Result")
print("========================================")

print(
    f"Maximum Epochs : {EPOCHS}"
)

print(
    f"Actual Epochs  : {actual_epochs}"
)


# =========================================================
# 13. 가장 좋은 Epoch 확인
# =========================================================

val_losses = history.history["val_loss"]

best_epoch = (
    np.argmin(val_losses) + 1
)

best_val_loss = (
    np.min(val_losses)
)

best_val_accuracy = (
    history.history["val_accuracy"][
        best_epoch - 1
    ]
)


print("\n========================================")
print("Best Epoch")
print("========================================")

print(
    f"Best Epoch        : {best_epoch}"
)

print(
    f"Best Val Accuracy : "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best Val Loss     : "
    f"{best_val_loss:.4f}"
)


# =========================================================
# 14. 저장된 Best Model 다시 불러오기
# =========================================================

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


# =========================================================
# 15. Best Model 최종 평가
# =========================================================

val_loss, val_accuracy = (
    best_model.evaluate(
        val_dataset,
        verbose=0
    )
)


print("\n========================================")
print("Best Model Evaluation")
print("========================================")

print(
    f"Validation Accuracy : "
    f"{val_accuracy:.4f}"
)

print(
    f"Validation Loss     : "
    f"{val_loss:.4f}"
)


# =========================================================
# 16. Accuracy 그래프
# =========================================================

epochs_range = range(
    1,
    actual_epochs + 1
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    epochs_range,
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

# Best Epoch 표시
plt.axvline(
    x=best_epoch,
    linestyle="--",
    label=f"Best Epoch ({best_epoch})"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "CNN EarlyStopping - "
    "Training and Validation Accuracy"
)

plt.legend()
plt.tight_layout()


accuracy_graph_path = (
    RESULTS_DIR
    / "cnn_earlystop_accuracy.png"
)

plt.savefig(
    accuracy_graph_path
)

plt.show()


# =========================================================
# 17. Loss 그래프
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    epochs_range,
    history.history["val_loss"],
    label="Validation Loss"
)

# Best Epoch 표시
plt.axvline(
    x=best_epoch,
    linestyle="--",
    label=f"Best Epoch ({best_epoch})"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "CNN EarlyStopping - "
    "Training and Validation Loss"
)

plt.legend()
plt.tight_layout()


loss_graph_path = (
    RESULTS_DIR
    / "cnn_earlystop_loss.png"
)

plt.savefig(
    loss_graph_path
)

plt.show()


# =========================================================
# 18. Best Model Validation 예측
# =========================================================

true_labels = []
predicted_labels = []


for images, labels in val_dataset:

    predictions = best_model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted_classes
    )


true_labels = np.array(
    true_labels
)

predicted_labels = np.array(
    predicted_labels
)


# =========================================================
# 19. Confusion Matrix
# =========================================================

confusion_matrix = (
    tf.math.confusion_matrix(
        true_labels,
        predicted_labels,
        num_classes=NUM_CLASSES
    ).numpy()
)


print("\n========================================")
print("Confusion Matrix")
print("========================================")

print(
    confusion_matrix
)


# =========================================================
# 20. Confusion Matrix 시각화
# =========================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    confusion_matrix,
    interpolation="nearest"
)

plt.title(
    "CNN EarlyStopping - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

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
    RESULTS_DIR
    / "cnn_earlystop_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path
)

plt.show()


# =========================================================
# 21. 클래스별 Accuracy
# =========================================================

print("\n========================================")
print("Accuracy by Class")
print("========================================")


for index, class_name in enumerate(
    class_names
):

    class_total = (
        confusion_matrix[index].sum()
    )

    class_correct = (
        confusion_matrix[index, index]
    )

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
# 22. 전체 Accuracy
# =========================================================

total_correct = np.trace(
    confusion_matrix
)

total_samples = (
    confusion_matrix.sum()
)

cm_accuracy = (
    total_correct / total_samples
)


print("\n========================================")
print("Confusion Matrix Accuracy")
print("========================================")

print(
    f"Correct : "
    f"{total_correct}/{total_samples}"
)

print(
    f"Accuracy: "
    f"{cm_accuracy:.4f}"
)


# =========================================================
# 23. Baseline 비교
# =========================================================

BASELINE_VAL_ACCURACY = 0.8917
BASELINE_VAL_LOSS = 0.2497


print("\n========================================")
print("Baseline vs EarlyStopping Best Model")
print("========================================")

print(
    f"Baseline Val Accuracy     : "
    f"{BASELINE_VAL_ACCURACY:.4f}"
)

print(
    f"EarlyStopping Val Accuracy: "
    f"{val_accuracy:.4f}"
)

print()

print(
    f"Baseline Val Loss         : "
    f"{BASELINE_VAL_LOSS:.4f}"
)

print(
    f"EarlyStopping Val Loss    : "
    f"{val_loss:.4f}"
)

print()

print(
    f"Best Epoch                : "
    f"{best_epoch}"
)

print(
    f"Actual Training Epochs    : "
    f"{actual_epochs}"
)