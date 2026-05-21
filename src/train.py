"""
train.py — Определение и обучение CNN-модели для классификации сетевых атак

Архитектура: 4 блока Conv2D → MaxPool → BatchNorm → Dropout
             GlobalAveragePooling → Dense → выходной слой

Запуск: python src/train.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    SPECTRO_DIR, MODELS_DIR, RESULTS_DIR,
    IMG_H, IMG_W, IMG_C,
    BATCH_SIZE, EPOCHS, LR, SEED,
    CLASSIFICATION_MODE, ATTACK_CLASSES,
    logger, set_seed, load_arrays, split_data, get_class_weights
)

# ─────────────────────────────────────────────
#  ПОСТРОЕНИЕ CNN
# ─────────────────────────────────────────────

def build_cnn(input_shape: tuple, n_classes: int) -> keras.Model:
    """
    Строит CNN для классификации спектрограмм.

    Параметры:
        input_shape : (H, W, C) — форма входного изображения
        n_classes   : 2 (binary) или 10 (multiclass)

    Архитектура:
        Вход → [Conv → BN → MaxPool → Dropout] x 4
             → GlobalAveragePooling
             → Dense(256) → Dropout
             → Dense(128) → Dropout
             → Выход (Sigmoid / Softmax)
    """
    inp = keras.Input(shape=input_shape, name="spectrogram_input")

    # ── Блок 1 ──────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # ── Блок 2 ──────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # ── Блок 3 ──────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.30, name="drop3")(x)

    # ── Блок 4 ──────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4), name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.40, name="drop4")(x)

    # ── Полносвязные слои ────────────────────
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.Dropout(0.40, name="drop5")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
    x = layers.Dropout(0.30, name="drop6")(x)

    # ── Выходной слой ────────────────────────
    if n_classes == 2:
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_Spectrogram_UNSWNB15")
    return model

# ─────────────────────────────────────────────
#  КОМПИЛЯЦИЯ
# ─────────────────────────────────────────────

def compile_model(model: keras.Model, n_classes: int) -> keras.Model:
    """Компилирует модель с нужной функцией потерь."""
    optimizer = keras.optimizers.Adam(learning_rate=LR)

    if n_classes == 2:
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy",
                     keras.metrics.AUC(name="auc"),
                     keras.metrics.Precision(name="precision"),
                     keras.metrics.Recall(name="recall")]
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy",
                     keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")]
        )
    logger.info("Модель скомпилирована")
    return model

# ─────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────

def get_callbacks(model_name: str) -> list:
    """Возвращает список callbacks для обучения."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.keras")

    return [
        # Сохранение лучшей модели
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Ранняя остановка
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Снижение learning rate при плато
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(RESULTS_DIR, "tb_logs"),
            histogram_freq=1
        )
    ]

# ─────────────────────────────────────────────
#  СОХРАНЕНИЕ ИСТОРИИ ОБУЧЕНИЯ
# ─────────────────────────────────────────────

def save_training_history(history, model_name: str):
    """Сохраняет историю обучения в JSON."""
    hist_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    hist_dict = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"История обучения сохранена: {hist_path}")

def plot_training_curves(history, model_name: str):
    """Строит и сохраняет графики accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Результаты обучения CNN (UNSW-NB15)\n"
        "Обнаружение сетевых атак по спектрограммам",
        fontsize=13, fontweight="bold"
    )

    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    # ── Accuracy ────────────────────────────
    ax1.plot(epochs_range, history.history["accuracy"],
             color="#2196F3", linewidth=2, label="Обучение")
    ax1.plot(epochs_range, history.history["val_accuracy"],
             color="#4CAF50", linewidth=2, linestyle="--", label="Валидация")
    ax1.set_xlabel("Эпоха", fontsize=11)
    ax1.set_ylabel("Точность (Accuracy)", fontsize=11)
    ax1.set_title("Точность по эпохам", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.35)
    ax1.set_ylim(0, 1.05)

    # Аннотация лучшей val_accuracy
    best_val_acc = max(history.history["val_accuracy"])
    best_ep      = history.history["val_accuracy"].index(best_val_acc) + 1
    ax1.annotate(
        f"Лучшая: {best_val_acc:.4f}\n(эп. {best_ep})",
        xy=(best_ep, best_val_acc),
        xytext=(best_ep + 1, best_val_acc - 0.05),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="darkgreen"
    )

    # ── Loss ────────────────────────────────
    ax2.plot(epochs_range, history.history["loss"],
             color="#F44336", linewidth=2, label="Обучение")
    ax2.plot(epochs_range, history.history["val_loss"],
             color="#FF9800", linewidth=2, linestyle="--", label="Валидация")
    ax2.set_xlabel("Эпоха", fontsize=11)
    ax2.set_ylabel("Потери (Loss)", fontsize=11)
    ax2.set_title("Потери по эпохам", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{model_name}_training_curves.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"График обучения сохранён: {out_path}")

# ─────────────────────────────────────────────
#  ОСНОВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def main():
    set_seed()
    logger.info("=" * 55)
    logger.info("  ЭТАП 3: ОБУЧЕНИЕ CNN")
    logger.info("=" * 55)

    # Выбор режима
    if CLASSIFICATION_MODE == "binary":
        prefix    = "spectrograms"
        n_classes = 2
        model_name = "cnn_binary"
    else:
        prefix    = "spectrograms"
        n_classes = len(ATTACK_CLASSES)
        model_name = "cnn_multiclass"

    # 1. Загрузка спектрограмм
    X, y = load_arrays(prefix)
    logger.info(f"X: {X.shape}, y: {y.shape}, классов: {n_classes}")

    # 2. Разбивка на train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Сохраняем тестовую выборку для evaluate.py
    np.save(os.path.join(SPECTRO_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(SPECTRO_DIR, "y_test.npy"), y_test)
    logger.info("Тестовая выборка сохранена")

    # 3. Веса классов (для дисбаланса)
    class_weights = get_class_weights(y_train)

    # 4. Построение модели
    input_shape = (IMG_H, IMG_W, IMG_C)
    model = build_cnn(input_shape, n_classes)
    model = compile_model(model, n_classes)
    model.summary(print_fn=logger.info)

    logger.info(f"Параметры модели: {model.count_params():,}")

    # 5. Обучение
    logger.info(f"Начало обучения: {EPOCHS} эпох, batch={BATCH_SIZE}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(model_name),
        verbose=1
    )

    # 6. Финальная оценка на валидации
    val_results = model.evaluate(X_val, y_val, verbose=0)
    logger.info("─" * 55)
    logger.info("  ИТОГИ ОБУЧЕНИЯ (на валидационной выборке):")
    for metric, val in zip(model.metrics_names, val_results):
        logger.info(f"    {metric:<20}: {val:.4f}")
    logger.info("─" * 55)

    # 7. Сохранение истории и графиков
    save_training_history(history, model_name)
    plot_training_curves(history, model_name)

    logger.info("Обучение завершено!")
    logger.info("Следующий шаг: python src/evaluate.py")


if __name__ == "__main__":
    main()
