"""
spectrogram_gen.py — Генерация спектрограмм из сетевых потоков UNSW-NB15

Принцип:
  Сетевые потоки сортируются по времени начала (Stime).
  Скользящим окном (WINDOW_SIZE потоков, шаг STRIDE) выбирается
  набор признаков, который формирует 2D-матрицу:
    строки  = числовые признаки (аналог частотной оси)
    столбцы = потоки во времени (аналог временной оси)
  Полученная матрица — спектрограмма сетевого трафика.
  Метка окна = класс большинства потоков внутри окна.

Запуск: python src/spectrogram_gen.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    DATA_PROC_DIR, SPECTRO_DIR, RESULTS_DIR,
    NUMERIC_FEATURES, WINDOW_SIZE, STRIDE, N_FEATURES,
    CLASSIFICATION_MODE, ATTACK_CLASSES,
    logger, set_seed, save_arrays
)

# ─────────────────────────────────────────────
#  ЗАГРУЗКА ОБРАБОТАННЫХ ДАННЫХ
# ─────────────────────────────────────────────

def load_processed(proc_dir: str = DATA_PROC_DIR) -> pd.DataFrame:
    """Загружает очищенный датафрейм."""
    path = os.path.join(proc_dir, "flows_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            "Сначала выполните: python src/preprocess.py"
        )
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Загружено потоков: {len(df):,}  из '{path}'")
    return df

# ─────────────────────────────────────────────
#  ПОДГОТОВКА МАССИВА ПРИЗНАКОВ
# ─────────────────────────────────────────────

def prepare_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Сортирует потоки по Stime и извлекает матрицу признаков.

    Возвращает:
        X_flows : (N, N_FEATURES) — нормализованные признаки каждого потока
        y_binary: (N,)            — бинарная метка (0=нормальный, 1=атака)
        y_multi : (N,)            — многоклассовая метка (0..9)
    """
    # Сортировка по времени
    if "Stime" in df.columns:
        df = df.sort_values("Stime").reset_index(drop=True)
        logger.info("Потоки отсортированы по Stime")

    # Проверка наличия всех признаков
    missing_cols = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют столбцы: {missing_cols}")

    X_flows = df[NUMERIC_FEATURES].values.astype(np.float32)   # (N, 32)
    y_binary = df["Label"].values.astype(np.int32)              # (N,)

    # Многоклассовые метки
    if "attack_cat_encoded" in df.columns:
        y_multi = df["attack_cat_encoded"].values.astype(np.int32)
    else:
        y_multi = y_binary.copy()

    logger.info(f"Матрица признаков: {X_flows.shape},  "
                f"Binary метки: {np.unique(y_binary)},  "
                f"Multi метки: {np.unique(y_multi)}")
    return X_flows, y_binary, y_multi

# ─────────────────────────────────────────────
#  ГЕНЕРАЦИЯ СПЕКТРОГРАММ (СКОЛЬЗЯЩЕЕ ОКНО)
# ─────────────────────────────────────────────

def generate_spectrograms(
        X_flows: np.ndarray,
        y_binary: np.ndarray,
        y_multi: np.ndarray,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Создаёт спектрограммы скользящим окном.

    Спектрограмма одного окна:
        shape = (N_FEATURES, window_size) = (32, 64)
        строки  → признаки (аналог частоты)
        столбцы → время (индекс потока в окне)
        значения → нормализованные признаки ∈ [0, 1]

    Возвращает:
        spectrograms : (M, 32, 64, 1)  — M спектрограмм
        labels_bin   : (M,)            — бинарные метки
        labels_multi : (M,)            — многоклассовые метки
    """
    N = len(X_flows)
    n_windows = (N - window_size) // stride + 1
    logger.info(f"Генерация спектрограмм: "
                f"{N:,} потоков → {n_windows:,} окон "
                f"(window={window_size}, stride={stride})")

    spectrograms  = np.zeros((n_windows, N_FEATURES, window_size, 1), dtype=np.float32)
    labels_bin    = np.zeros(n_windows, dtype=np.int32)
    labels_multi  = np.zeros(n_windows, dtype=np.int32)

    for i in tqdm(range(n_windows), desc="Генерация спектрограмм", unit="окно"):
        start = i * stride
        end   = start + window_size

        # Окно признаков: (window_size, N_FEATURES)
        window_data = X_flows[start:end]          # (64, 32)

        # Транспонируем: (N_FEATURES, window_size) = (32, 64)
        spectrogram = window_data.T               # (32, 64)

        # Логарифмическое масштабирование (усиливает малые значения, как в STFT)
        spectrogram = np.log1p(spectrogram * 10.0)

        # Локальная нормализация [0, 1]
        s_min, s_max = spectrogram.min(), spectrogram.max()
        if s_max > s_min:
            spectrogram = (spectrogram - s_min) / (s_max - s_min)

        spectrograms[i, :, :, 0] = spectrogram

        # Метка окна = мажоритарный класс
        window_bin   = y_binary[start:end]
        window_multi = y_multi[start:end]
        # labels_bin[i]   = np.bincount(window_bin).argmax()
      # Option B — any-attack labeling
    if np.any(window_bin == 1):
        labels_bin[i] = 1
    else:
        labels_bin[i] = 0
    
    if np.any(window_bin == 1):
                attack_labels = window_multi[window_bin == 1]
                labels_multi[i] = np.bincount(attack_labels).argmax()
            else:
                labels_multi[i] = 7  # Normal
        logger.info(f"Итого спектрограмм: {len(spectrograms):,}  "
                f"(форма: {spectrograms.shape})")

    # Статистика по меткам
    unique_b, counts_b = np.unique(labels_bin, return_counts=True)
    logger.info("  Бинарное распределение меток:")
    for cls, cnt in zip(unique_b, counts_b):
        logger.info(f"    {'Нормальный' if cls==0 else 'Атака':>12}: {cnt:>7,}  "
                    f"({cnt/len(labels_bin)*100:.1f}%)")

    return spectrograms, labels_bin, labels_multi

# ─────────────────────────────────────────────
#  ВИЗУАЛИЗАЦИЯ ПРИМЕРОВ
# ─────────────────────────────────────────────

def visualize_samples(
        spectrograms: np.ndarray,
        labels_bin: np.ndarray,
        labels_multi: np.ndarray,
        encoders: dict = None,
        n_samples: int = 8,
        out_dir: str = RESULTS_DIR
):
    """
    Сохраняет сетку из примеров спектрограмм (нормальный / атаки).
    """
    logger.info("Визуализация примеров спектрограмм...")

    # Индексы: по 4 нормальных и 4 атаки
    idx_norm = np.where(labels_bin == 0)[0]
    idx_atk  = np.where(labels_bin == 1)[0]

    n_each = n_samples // 2
    chosen = np.concatenate([
        idx_norm[:n_each],
        idx_atk[:n_each]
    ])

    fig, axes = plt.subplots(2, n_each, figsize=(16, 6))
    fig.suptitle(
        "Примеры спектрограмм сетевого трафика UNSW-NB15\n"
        "(строки = признаки, столбцы = потоки во времени)",
        fontsize=13
    )

    for col_i, idx in enumerate(chosen):
        row = 0 if col_i < n_each else 1
        col = col_i % n_each
        ax  = axes[row, col]

        spec = spectrograms[idx, :, :, 0]
        ax.imshow(spec, aspect="auto", origin="lower",
                  cmap="viridis", vmin=0, vmax=1)

        label_name = "Нормальный" if labels_bin[idx] == 0 else "Атака"
        ax.set_title(f"{label_name}", fontsize=9, fontweight="bold",
                     color="steelblue" if labels_bin[idx] == 0 else "crimson")
        ax.set_xlabel("Время (потоки)", fontsize=7)
        ax.set_ylabel("Признак (частота)", fontsize=7)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "spectrogram_samples.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранён пример спектрограмм: {out_path}")

def visualize_per_attack_class(
        spectrograms: np.ndarray,
        labels_multi: np.ndarray,
        encoders: dict,
        out_dir: str = RESULTS_DIR
):
    """Одна спектрограмма на каждый класс атак."""
    if "attack_cat" not in encoders:
        return

    class_names = encoders["attack_cat"].classes_
    n_classes   = len(class_names)
    cols = min(n_classes, 5)
    rows = (n_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).flatten()

    fig.suptitle(
        "Спектрограммы по классам атак (UNSW-NB15)",
        fontsize=14, fontweight="bold"
    )

    for cls_id, cls_name in enumerate(class_names):
        idxs = np.where(labels_multi == cls_id)[0]
        ax   = axes[cls_id]
        if len(idxs) == 0:
            ax.axis("off")
            ax.set_title(f"{cls_name}\n(нет данных)", fontsize=8)
            continue
        spec = spectrograms[idxs[0], :, :, 0]
        ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(cls_name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Время", fontsize=7)
        ax.set_ylabel("Признак", fontsize=7)
        ax.tick_params(labelsize=6)

    # Скрываем пустые оси
    for i in range(n_classes, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "spectrogram_per_class.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранён: {out_path}")

# ─────────────────────────────────────────────
#  ОСНОВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def main():
    import pickle

    set_seed()
    logger.info("=" * 55)
    logger.info("  ЭТАП 2: ГЕНЕРАЦИЯ СПЕКТРОГРАММ")
    logger.info("=" * 55)

    # Загрузка энкодеров
    enc_path = os.path.join(DATA_PROC_DIR, "label_encoder.pkl")
    encoders = {}
    if os.path.exists(enc_path):
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)

    # 1. Загрузка данных
    df = load_processed()

    # 2. Матрица признаков
    X_flows, y_binary, y_multi = prepare_feature_matrix(df)

    # 3. Генерация спектрограмм
    spectrograms, labels_bin, labels_multi = generate_spectrograms(
        X_flows, y_binary, y_multi
    )

    # 4. Сохранение (бинарные метки — основной режим)
    if CLASSIFICATION_MODE == "binary":
        save_arrays(spectrograms, labels_bin,   prefix="spectrograms")
        save_arrays(spectrograms, labels_multi, prefix="spectrograms_multi")
    else:
        save_arrays(spectrograms, labels_multi, prefix="spectrograms")
        save_arrays(spectrograms, labels_bin,   prefix="spectrograms_binary")

    logger.info(f"Спектрограммы сохранены в: {SPECTRO_DIR}")

    # 5. Визуализация
    visualize_samples(spectrograms, labels_bin, labels_multi, encoders)
    visualize_per_attack_class(spectrograms, labels_multi, encoders)

    logger.info("Генерация спектрограмм завершена!")
    logger.info("Следующий шаг: python src/train.py")


if __name__ == "__main__":
    main()
