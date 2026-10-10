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
    MODEL_DIR / "mobilenetv2_transfer_best.keras"
)


# =========================================================
# 2. 기본 설정
# =========================================================

IMAGE_SIZE = (200, 200)
BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 6

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
# 8. MobileNetV2 Base Model 불러오기
# =========================================================

base_model = (
    tf.keras.applications.MobileNetV2(
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),

        # 기존 ImageNet 분류기 제거
        include_top=False,

        # ImageNet 사전학습 가중치 사용
        weights="imagenet"
    )
)


# =========================================================
# 9. Base Model 동결
# =========================================================

base_model.trainable = False


print("\n========================================")
print("MobileNetV2 Base Model")
print("========================================")

print(
    f"Base Model Trainable : "
    f"{base_model.trainable}"
)

print(
    f"Number of Layers     : "
    f"{len(base_model.layers)}"
)


# =========================================================
# 10. Transfer Learning 모델 생성
# =========================================================

inputs = tf.keras.Input(
    shape=(200, 200, 3)
)


# MobileNetV2 전처리
x = tf.keras.applications.mobilenet_v2.preprocess_input(
    inputs
)


# 사전학습된 Feature Extractor
x = base_model(
    x,
    training=False
)


# Feature Map → 하나의 Feature Vector
x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)


# NEU-DET용 분류기
x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(
    x
)


# 최종 6개 클래스 출력
outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(
    x
)


model = tf.keras.Model(
    inputs,
    outputs
)


# =========================================================
# 11. 모델 구조 확인
# =========================================================

print("\n========================================")
print("MobileNetV2 Transfer Learning Model")
print("========================================")

model.summary()


# =========================================================
# 12. Trainable Parameter 확인
# =========================================================

trainable_parameters = np.sum([
    np.prod(variable.shape)
    for variable in model.trainable_weights
])

non_trainable_parameters = np.sum([
    np.prod(variable.shape)
    for variable in model.non_trainable_weights
])


print("\n========================================")
print("Parameter Information")
print("========================================")

print(
    f"Trainable Parameters     : "
    f"{trainable_parameters:,}"
)

print(
    f"Non-trainable Parameters : "
    f"{non_trainable_parameters:,}"
)


# =========================================================
# 13. 모델 Compile
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# =========================================================
# 14. Callback 설정
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
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
)


# =========================================================
# 15. Transfer Learning 시작
# =========================================================

print("\n========================================")
print("Transfer Learning Start")
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
# 16. 실제 학습 Epoch 확인
# =========================================================

actual_epochs = len(
    history.history["loss"]
)


print("\n========================================")
print("Training Result")
print("========================================")

print(
    f"Maximum Epochs : {EPOCHS}"
)

print(
    f"Actual Epochs  : {actual_epochs}"
)


# =========================================================
# 17. Best Epoch 확인
# =========================================================

val_losses = history.history[
    "val_loss"
]

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
# 18. Best Model 불러오기
# =========================================================

best_model = (
    tf.keras.models.load_model(
        BEST_MODEL_PATH
    )
)


# =========================================================
# 19. Best Model 평가
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
# 20. Accuracy 그래프
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
    "MobileNetV2 Transfer Learning - Accuracy"
)

plt.legend()
plt.tight_layout()


accuracy_graph_path = (
    RESULTS_DIR
    / "mobilenetv2_transfer_accuracy.png"
)

plt.savefig(
    accuracy_graph_path
)

plt.show()


# =========================================================
# 21. Loss 그래프
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
    "MobileNetV2 Transfer Learning - Loss"
)

plt.legend()
plt.tight_layout()


loss_graph_path = (
    RESULTS_DIR
    / "mobilenetv2_transfer_loss.png"
)

plt.savefig(
    loss_graph_path
)

plt.show()


# =========================================================
# 22. Validation 데이터 예측
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
# 23. Confusion Matrix 생성
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
# 24. Confusion Matrix 시각화
# =========================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    confusion_matrix,
    interpolation="nearest"
)

plt.title(
    "MobileNetV2 Transfer Learning - Confusion Matrix"
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
    / "mobilenetv2_transfer_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path
)

plt.show()


# =========================================================
# 25. 클래스별 Accuracy
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
# 26. Confusion Matrix 전체 Accuracy
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
# 27. 기존 CNN Best Model과 비교
# =========================================================

CNN_BEST_VAL_ACCURACY = 0.9056
CNN_BEST_VAL_LOSS = 0.2201


print("\n========================================")
print("CNN Best vs MobileNetV2 Transfer")
print("========================================")

print(
    f"CNN Best Val Accuracy       : "
    f"{CNN_BEST_VAL_ACCURACY:.4f}"
)

print(
    f"MobileNetV2 Val Accuracy    : "
    f"{val_accuracy:.4f}"
)

print()

print(
    f"CNN Best Val Loss           : "
    f"{CNN_BEST_VAL_LOSS:.4f}"
)

print(
    f"MobileNetV2 Val Loss        : "
    f"{val_loss:.4f}"
)

print()

print(
    f"Transfer Learning Best Epoch: "
    f"{best_epoch}"
)

print(
    f"Actual Training Epochs      : "
    f"{actual_epochs}"
)