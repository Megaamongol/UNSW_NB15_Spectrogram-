"""
preprocess.py — Загрузка, очистка и предобработка датасета UNSW-NB15
Выход: data/processed/flows_clean.csv  +  data/processed/label_encoder.pkl

Запуск: python src/preprocess.py
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    DATA_RAW_DIR, DATA_PROC_DIR, NUMERIC_FEATURES,
    ATTACK_CLASSES, CLASSIFICATION_MODE, logger, set_seed, SEED
)

# ─────────────────────────────────────────────
#  НАЗВАНИЯ СТОЛБЦОВ UNSW-NB15
# ─────────────────────────────────────────────
UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "Label"
]

# Категориальные признаки (будут закодированы)
CATEGORICAL_FEATURES = ["proto", "state", "service"]

# ─────────────────────────────────────────────
#  ЗАГРУЗКА CSV
# ─────────────────────────────────────────────

def load_raw_data(raw_dir: str = DATA_RAW_DIR) -> pd.DataFrame:
    """
    Загружает все CSV-файлы UNSW-NB15 из папки raw/.
    Поддерживает файлы как со заголовком, так и без него.
    """
    csv_files = [os.path.join(raw_dir, "UNSW-NB15_merged.csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"CSV-файлы не найдены в '{raw_dir}'.\n"
            "Поместите UNSW-NB15_1.csv ... UNSW-NB15_4.csv в папку data/raw/"
        )
    logger.info(f"Найдено CSV-файлов: {len(csv_files)}")

    frames = []
    for path in csv_files:
        fname = os.path.basename(path)
        # Проверяем наличие заголовка
        first_row = pd.read_csv(path, nrows=1, header=None).iloc[0].tolist()
        has_header = isinstance(first_row[0], str) and first_row[0].lower() in [
            "srcip", "source ip", "src_ip"
        ]
        if has_header:
            df = pd.read_csv(path, low_memory=False)
        else:
            df = pd.read_csv(path, header=None, low_memory=False)
            if df.shape[1] == len(UNSW_COLUMNS):
                df.columns = UNSW_COLUMNS
            elif df.shape[1] == len(UNSW_COLUMNS) - 1:
                # Некоторые версии без attack_cat
                df.columns = UNSW_COLUMNS[:-2] + ["Label"]
                df["attack_cat"] = "Unknown"
            else:
                logger.warning(
                    f"{fname}: неожиданное кол-во столбцов ({df.shape[1]}), пропускаем"
                )
                continue

        frames.append(df)
        logger.info(f"  {fname}: {len(df):,} строк")

    data = pd.concat(frames, ignore_index=True)
    logger.info(f"Итого загружено: {len(data):,} потоков")
    return data

# ─────────────────────────────────────────────
#  ОЧИСТКА
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает датафрейм:
    - Удаляет дубликаты
    - Обрабатывает пропуски
    - Исправляет типы данных
    - Стандартизирует attack_cat
    """
    n_before = len(df)

    # Удаление полных дубликатов
    df = df.drop_duplicates()
    logger.info(f"Удалено дубликатов: {n_before - len(df):,}")

    # Стандартизация attack_cat: убираем пробелы, приводим к Title Case
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip().str.title()
        df["attack_cat"] = df["attack_cat"].replace({
            "Normal": "Normal",
            "Nan": "Normal",
            "": "Normal",
        })
        # Принудительно: если Label==0, то Normal
        df.loc[df["Label"] == 0, "attack_cat"] = "Normal"

    # Числовые признаки: заменяем нечисловые значения на NaN, затем медиану
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Заполнение пропусков медианой по числовым признакам
    missing_before = df[NUMERIC_FEATURES].isnull().sum().sum()
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(
        df[NUMERIC_FEATURES].median(numeric_only=True)
    )
    logger.info(f"Заполнено пропусков (медиана): {missing_before:,}")

    # Категориальные признаки: заменяем пропуски на '-'
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("-").astype(str).str.strip().str.lower()

    # Метка: убеждаемся, что Label — целое 0/1
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    df["Label"] = df["Label"].clip(0, 1)

    # Stime: числовой (Unix timestamp)
    if "Stime" in df.columns:
        df["Stime"] = pd.to_numeric(df["Stime"], errors="coerce")
        df = df.dropna(subset=["Stime"])

    logger.info(f"После очистки: {len(df):,} потоков")
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────
#  КОДИРОВАНИЕ ПРИЗНАКОВ
# ─────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Кодирует категориальные признаки (LabelEncoding).
    Возвращает датафрейм и словарь энкодеров.
    """
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"  Закодировано '{col}': {len(le.classes_)} уникальных значений")

    # Кодирование attack_cat для многоклассовой задачи
    le_attack = LabelEncoder()
    df["attack_cat_encoded"] = le_attack.fit_transform(df["attack_cat"].astype(str))
    encoders["attack_cat"] = le_attack
    logger.info(f"  Классы атак: {list(le_attack.classes_)}")

    return df, encoders

# ─────────────────────────────────────────────
#  НОРМАЛИЗАЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ
# ─────────────────────────────────────────────

def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """MinMax нормализация числовых признаков в диапазон [0, 1]."""
    scaler = MinMaxScaler()
    df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])
    logger.info(f"MinMax нормализация применена к {len(NUMERIC_FEATURES)} признакам")
    return df, scaler

# ─────────────────────────────────────────────
#  СОХРАНЕНИЕ
# ─────────────────────────────────────────────

def save_processed(df: pd.DataFrame, encoders: dict, scaler: MinMaxScaler,
                   out_dir: str = DATA_PROC_DIR):
    """Сохраняет обработанные данные и вспомогательные объекты."""
    csv_path = os.path.join(out_dir, "flows_clean.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Сохранён: {csv_path}  ({len(df):,} строк)")

    enc_path = os.path.join(out_dir, "label_encoder.pkl")
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)
    logger.info(f"Сохранён: {enc_path}")

    scaler_path = os.path.join(out_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Сохранён: {scaler_path}")

# ─────────────────────────────────────────────
#  СТАТИСТИКА ДАТАСЕТА
# ─────────────────────────────────────────────

def print_dataset_stats(df: pd.DataFrame):
    """Выводит статистику по классам."""
    logger.info("─" * 55)
    logger.info("  СТАТИСТИКА ДАТАСЕТА")
    logger.info("─" * 55)

    # Бинарная статистика
    counts = df["Label"].value_counts().sort_index()
    total = len(df)
    logger.info(f"  Нормальный трафик : {counts.get(0, 0):>8,}  "
                f"({counts.get(0, 0)/total*100:.1f}%)")
    logger.info(f"  Атаки             : {counts.get(1, 0):>8,}  "
                f"({counts.get(1, 0)/total*100:.1f}%)")
    logger.info(f"  ИТОГО             : {total:>8,}")

    # По типам атак
    logger.info("\n  Распределение по типам атак:")
    cat_counts = df["attack_cat"].value_counts()
    for cat, cnt in cat_counts.items():
        logger.info(f"    {cat:<20}: {cnt:>7,}  ({cnt/total*100:.2f}%)")
    logger.info("─" * 55)

# ─────────────────────────────────────────────
#  ОСНОВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def main():
    set_seed()
    logger.info("=" * 55)
    logger.info("  ЭТАП 1: ПРЕДОБРАБОТКА ДАННЫХ UNSW-NB15")
    logger.info("=" * 55)

    # 1. Загрузка
    df = load_raw_data()

    # 2. Очистка
    df = clean_data(df)

    # 3. Кодирование
    df, encoders = encode_features(df)

    # 4. Нормализация
    df, scaler = normalize_features(df)

    # 5. Статистика
    print_dataset_stats(df)

    # 6. Сохранение
    save_processed(df, encoders, scaler)

    logger.info("Предобработка завершена успешно!")
    logger.info(f"Следующий шаг: python src/spectrogram_gen.py")


if __name__ == "__main__":
    main()
