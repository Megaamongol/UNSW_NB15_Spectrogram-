"""
evaluate.py — Оценка обученной CNN-модели на тестовой выборке

Генерирует:
  - Матрицу ошибок (Confusion Matrix)
  - Отчёт по классификации (Precision / Recall / F1)
  - ROC-кривую (для бинарной задачи)
  - Итоговую таблицу метрик (для главы 3 диплома)

Запуск: python src/evaluate.py
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    SPECTRO_DIR, MODELS_DIR, RESULTS_DIR, DATA_PROC_DIR,
    CLASSIFICATION_MODE, ATTACK_CLASSES,
    logger, set_seed, save_log_to_txt
)

# ─────────────────────────────────────────────
#  ЗАГРУЗКА МОДЕЛИ И ТЕСТОВЫХ ДАННЫХ
# ─────────────────────────────────────────────

def load_model_and_data(mode: str = CLASSIFICATION_MODE):
    """Загружает сохранённую модель и тестовые данные."""
    model_name = "cnn_binary" if mode == "binary" else "cnn_multiclass"
    model_path = os.path.join(MODELS_DIR, f"{model_name}.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}\n"
            "Сначала выполните: python src/train.py"
        )

    model = keras.models.load_model(model_path)
    logger.info(f"Модель загружена: {model_path}")

    X_test = np.load(os.path.join(SPECTRO_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(SPECTRO_DIR, "y_test.npy"))
    logger.info(f"Тестовая выборка: X{X_test.shape}, y{y_test.shape}")

    return model, X_test, y_test, model_name

def load_encoders() -> dict:
    """Загружает словарь LabelEncoder."""
    enc_path = os.path.join(DATA_PROC_DIR, "label_encoder.pkl")
    if os.path.exists(enc_path):
        with open(enc_path, "rb") as f:
            return pickle.load(f)
    return {}

# ─────────────────────────────────────────────
#  ПРЕДСКАЗАНИЯ
# ─────────────────────────────────────────────

def get_predictions(model, X_test: np.ndarray, n_classes: int):
    """
    Возвращает:
        y_pred_prob : вероятности
        y_pred      : предсказанные классы
    """
    y_pred_prob = model.predict(X_test, batch_size=64, verbose=0)

    if n_classes == 2:
        y_pred      = (y_pred_prob.squeeze() >= 0.5).astype(int)
        y_pred_prob = y_pred_prob.squeeze()
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)

    return y_pred_prob, y_pred

# ─────────────────────────────────────────────
#  CONFUSION MATRIX
# ─────────────────────────────────────────────

def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list,
        model_name: str,
        out_dir: str = RESULTS_DIR
):
    """Строит и сохраняет нормированную матрицу ошибок."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(8, n * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Предсказанный класс", fontsize=11)
    ax.set_ylabel("Истинный класс",      fontsize=11)
    ax.set_title(
        "Матрица ошибок (Confusion Matrix)\n"
        "CNN — обнаружение сетевых атак по спектрограммам (UNSW-NB15)",
        fontsize=12, fontweight="bold"
    )

    for i in range(n):
        for j in range(n):
            val    = cm_norm[i, j]
            count  = cm[i, j]
            color  = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}\n({count:,})",
                    ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Матрица ошибок сохранена: {out_path}")
    return cm

# ─────────────────────────────────────────────
#  ROC-КРИВАЯ (только для бинарной задачи)
# ─────────────────────────────────────────────

def plot_roc_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        out_dir: str = RESULTS_DIR
):
    """Строит ROC-кривую и сохраняет график."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
            label=f"CNN (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5,
            label="Случайный классификатор (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#2196F3")

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title(
        "ROC-кривая — CNN классификатор\n"
        "Обнаружение сетевых атак по спектрограммам (UNSW-NB15)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_roc_curve.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC-кривая сохранена: {out_path}  (AUC = {roc_auc:.4f})")
    return roc_auc

# ─────────────────────────────────────────────
#  ИТОГОВЫЕ МЕТРИКИ
# ─────────────────────────────────────────────

def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob,
        n_classes: int,
        class_names: list
) -> dict:
    """Вычисляет все метрики и возвращает словарь."""
    avg = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1_score" : f1_score(y_true, y_pred, average=avg, zero_division=0),
    }

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["auc"] = float(auc(fpr, tpr))

    # Подробный отчёт
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )
    metrics["classification_report"] = report

    return metrics

def print_metrics_table(metrics: dict, class_names: list):
    """Выводит итоговую таблицу метрик в лог."""
    logger.info("=" * 60)
    logger.info("  ИТОГОВЫЕ МЕТРИКИ  (Глава 3 диплома)")
    logger.info("=" * 60)
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}  "
                f"({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1-Score  : {metrics['f1_score']:.4f}")
    if "auc" in metrics:
        logger.info(f"  AUC-ROC   : {metrics['auc']:.4f}")
    logger.info("─" * 60)
    logger.info("  Детализация по классам:")
    report = metrics["classification_report"]
    for cls in class_names:
        if cls in report:
            r = report[cls]
            logger.info(
                f"    {cls:<20} P={r['precision']:.3f}  "
                f"R={r['recall']:.3f}  F1={r['f1-score']:.3f}  "
                f"N={int(r['support'])}"
            )
    logger.info("=" * 60)

def plot_metrics_bar(metrics: dict, model_name: str, out_dir: str = RESULTS_DIR):
    """Гистограмма основных метрик."""
    keys   = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    bars   = ax.bar(labels, values, color=colors, edgecolor="white",
                    linewidth=1.5, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Значение метрики", fontsize=12)
    ax.set_title(
        "Метрики качества CNN-классификатора\n"
        "Обнаружение сетевых атак по спектрограммам (UNSW-NB15)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name}_metrics_bar.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"График метрик сохранён: {out_path}")

def save_metrics_json(metrics: dict, model_name: str, out_dir: str = RESULTS_DIR):
    """Сохраняет метрики в JSON для диплома."""
    path = os.path.join(out_dir, f"{model_name}_metrics.json")
    exportable = {k: v for k, v in metrics.items()
                  if not isinstance(v, dict)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exportable, f, indent=2, ensure_ascii=False)
    logger.info(f"Метрики сохранены: {path}")

# ─────────────────────────────────────────────
#  ОСНОВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def main():
    set_seed()
    logger.info("=" * 60)
    logger.info("  ЭТАП 4: ОЦЕНКА МОДЕЛИ")
    logger.info("=" * 60)

    encoders  = load_encoders()
    n_classes = 2 if CLASSIFICATION_MODE == "binary" else len(ATTACK_CLASSES)

    # Имена классов
    if CLASSIFICATION_MODE == "binary":
        class_names = ["Нормальный", "Атака"]
    else:
        if "attack_cat" in encoders:
            class_names = list(encoders["attack_cat"].classes_)
        else:
            class_names = ATTACK_CLASSES

    # 1. Загрузка модели и данных
    model, X_test, y_test, model_name = load_model_and_data()

    # 2. Предсказания
    y_prob, y_pred = get_predictions(model, X_test, n_classes)

    # 3. Метрики
    metrics = compute_metrics(y_test, y_pred, y_prob, n_classes, class_names)
    print_metrics_table(metrics, class_names)
    save_metrics_json(metrics, model_name)

    # 4. Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, class_names, model_name)

    # 5. ROC-кривая (только binary)
    if n_classes == 2:
        plot_roc_curve(y_test, y_prob, model_name)

    # 6. Гистограмма метрик
    plot_metrics_bar(metrics, model_name)

    logger.info("─" * 60)
    logger.info("  ВСЕ РЕЗУЛЬТАТЫ СОХРАНЕНЫ В папке: results/")
    logger.info("─" * 60)
    logger.info(
        f"  Итог: Accuracy={metrics['accuracy']:.4f}, "
        f"F1={metrics['f1_score']:.4f}"
        + (f", AUC={metrics['auc']:.4f}" if 'auc' in metrics else "")
    )
    logger.info("Оценка модели завершена!")
    
    #7. resultat txt
    logger.info("Оценка модели завершена!")
    save_log_to_txt("04_evaluate")     


if __name__ == "__main__":
    main()
