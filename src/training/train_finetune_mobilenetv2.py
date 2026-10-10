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

TRANSFER_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_transfer_best.keras"
)

FINE_TUNE_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_finetune_best.keras"
)


# =========================================================
# 2. 기본 설정
# =========================================================

IMAGE_SIZE = (200, 200)
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 6

EPOCHS = 20

FINE_TUNE_LAYERS = 30
FINE_TUNE_LEARNING_RATE = 1e-5


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
# 8. Transfer Learning Best Model 불러오기
# =========================================================

model = tf.keras.models.load_model(
    TRANSFER_MODEL_PATH
)


print("\n========================================")
print("Loaded Transfer Learning Model")
print("========================================")

print(
    f"Loaded model: "
    f"{TRANSFER_MODEL_PATH}"
)


# =========================================================
# 9. MobileNetV2 Base Model 찾기
# =========================================================

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break


if base_model is None:

    raise ValueError(
        "MobileNetV2 base model을 찾을 수 없습니다."
    )


print("\n========================================")
print("Base Model Information")
print("========================================")

print(
    f"Base Model Name   : "
    f"{base_model.name}"
)

print(
    f"Number of Layers  : "
    f"{len(base_model.layers)}"
)


# =========================================================
# 10. Base Model 일부 레이어 Fine-tuning
# =========================================================

base_model.trainable = True


# 앞부분은 다시 동결
for layer in base_model.layers[
    :-FINE_TUNE_LAYERS
]:
    layer.trainable = False


# 마지막 30개 레이어 중
# BatchNormalization은 안정성을 위해 동결
for layer in base_model.layers[
    -FINE_TUNE_LAYERS:
]:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


# =========================================================
# 11. Trainable Layer 확인
# =========================================================

trainable_layer_count = sum(
    1
    for layer in base_model.layers
    if layer.trainable
)


print("\n========================================")
print("Fine-tuning Layer Information")
print("========================================")

print(
    f"Fine-tune target layers : "
    f"{FINE_TUNE_LAYERS}"
)

print(
    f"Trainable base layers   : "
    f"{trainable_layer_count}"
)


# =========================================================
# 12. 모델 재 Compile
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINE_TUNE_LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("\n========================================")
print("Fine-tuning Compile Complete")
print("========================================")

print(
    f"Learning Rate : "
    f"{FINE_TUNE_LEARNING_RATE}"
)


# =========================================================
# 13. Callback 설정
# =========================================================

early_stopping = (
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
)


model_checkpoint = (
    tf.keras.callbacks.ModelCheckpoint(
        filepath=FINE_TUNE_MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
)


# =========================================================
# 14. Fine-tuning 시작
# =========================================================

print("\n========================================")
print("Fine-tuning Start")
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
# 15. 실제 학습 Epoch 확인
# =========================================================

actual_epochs = len(
    history.history["loss"]
)


print("\n========================================")
print("Fine-tuning Result")
print("========================================")

print(
    f"Maximum Epochs : "
    f"{EPOCHS}"
)

print(
    f"Actual Epochs  : "
    f"{actual_epochs}"
)


# =========================================================
# 16. Best Epoch 확인
# =========================================================

val_losses = (
    history.history["val_loss"]
)

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
    f"Best Epoch        : "
    f"{best_epoch}"
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
# 17. Fine-tuning Best Model 불러오기
# =========================================================

best_model = (
    tf.keras.models.load_model(
        FINE_TUNE_MODEL_PATH
    )
)


# =========================================================
# 18. Best Model 평가
# =========================================================

val_loss, val_accuracy = (
    best_model.evaluate(
        val_dataset,
        verbose=0
    )
)


print("\n========================================")
print("Fine-tuning Best Model Evaluation")
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
# 19. Accuracy 그래프
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
    "MobileNetV2 Fine-tuning - Accuracy"
)

plt.legend()
plt.tight_layout()


accuracy_graph_path = (
    RESULTS_DIR
    / "mobilenetv2_finetune_accuracy.png"
)

plt.savefig(
    accuracy_graph_path
)

plt.show()


# =========================================================
# 20. Loss 그래프
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
    "MobileNetV2 Fine-tuning - Loss"
)

plt.legend()
plt.tight_layout()


loss_graph_path = (
    RESULTS_DIR
    / "mobilenetv2_finetune_loss.png"
)

plt.savefig(
    loss_graph_path
)

plt.show()


# =========================================================
# 21. Validation 예측
# =========================================================

true_labels = []
predicted_labels = []


for images, labels in val_dataset:

    predictions = (
        best_model.predict(
            images,
            verbose=0
        )
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
# 22. Confusion Matrix
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
# 23. Confusion Matrix 시각화
# =========================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    confusion_matrix,
    interpolation="nearest"
)

plt.title(
    "MobileNetV2 Fine-tuning - Confusion Matrix"
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
    / "mobilenetv2_finetune_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path
)

plt.show()


# =========================================================
# 24. 클래스별 Accuracy
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
# 25. 전체 Accuracy
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
# 26. Transfer vs Fine-tuning 비교
# =========================================================

TRANSFER_VAL_ACCURACY = 0.9972
TRANSFER_VAL_LOSS = 0.0074


print("\n========================================")
print("Transfer Learning vs Fine-tuning")
print("========================================")

print(
    f"Transfer Val Accuracy : "
    f"{TRANSFER_VAL_ACCURACY:.4f}"
)

print(
    f"Fine-tune Val Accuracy: "
    f"{val_accuracy:.4f}"
)

print()

print(
    f"Transfer Val Loss     : "
    f"{TRANSFER_VAL_LOSS:.4f}"
)

print(
    f"Fine-tune Val Loss    : "
    f"{val_loss:.4f}"
)

print()

print(
    f"Fine-tune Best Epoch  : "
    f"{best_epoch}"
)

print(
    f"Actual Training Epochs: "
    f"{actual_epochs}"
)