"""
utils.py — Вспомогательные функции и конфигурация проекта
Тема диплома: Оценка наличия сетевых атак по спектрограммам в компьютерных сетях
Датасет: UNSW-NB15
"""

import os
import logging
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
#  ПУТИ
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
SPECTRO_DIR     = os.path.join(BASE_DIR, "spectrograms")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

for _d in [DATA_RAW_DIR, DATA_PROC_DIR, SPECTRO_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────
#  КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────

# Режим классификации: "binary" (нормальный / атака)  или  "multiclass" (10 классов)
CLASSIFICATION_MODE = "binary"

# Параметры спектрограммы
WINDOW_SIZE  = 64      # кол-во потоков в одном окне (ширина спектрограммы)
STRIDE       = 32      # шаг сдвига скользящего окна
N_FEATURES   = 32      # кол-во числовых признаков (высота спектрограммы)
IMG_H        = N_FEATURES   # высота изображения
IMG_W        = WINDOW_SIZE  # ширина изображения
IMG_C        = 1            # каналы (grayscale)

# Параметры обучения
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-3
SEED         = 42

# Размер тестовой и валидационной выборок (доля)
VAL_SIZE     = 0.15
TEST_SIZE    = 0.15

# Классы для многоклассовой классификации
ATTACK_CLASSES = [
    "Normal", "Fuzzers", "Analysis", "Backdoors",
    "DoS", "Exploits", "Generic", "Reconnaissance",
    "Shellcode", "Worms"
]

# Числовые признаки, используемые для построения спектрограммы
NUMERIC_FEATURES = [
    "dur", "sbytes", "dbytes", "sttl", "dttl",
    "sloss", "dloss", "Sload", "Dload", "Spkts",
    "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Sintpkt", "Dintpkt", "tcprtt",
    "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm"
]

# Убеждаемся, что ровно N_FEATURES признаков
assert len(NUMERIC_FEATURES) == N_FEATURES, (
    f"Ожидается {N_FEATURES} признаков, найдено {len(NUMERIC_FEATURES)}"
)

# ─────────────────────────────────────────────
#  ЛОГИРОВАНИЕ
# ─────────────────────────────────────────────

def get_logger(name: str = "unsw_nb15") -> logging.Logger:
    """Возвращает настроенный логгер."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Вывод в консоль
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # Вывод в файл
        fh = logging.FileHandler(os.path.join(BASE_DIR, "pipeline.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = get_logger()

# ─────────────────────────────────────────────
#  INTERACTIVE RESEARCH SETTINGS
# ─────────────────────────────────────────────

SAVE_LOG = input("\nСудалгааны лог TXT файлд хадгалах уу? (y/n): ").strip().lower()

if SAVE_LOG == "y":
    log_name = input("TXT файлын нэр оруулна уу: ").strip()

    if not log_name.endswith(".txt"):
        log_name += ".txt"

    txt_dir = os.path.join(RESULTS_DIR, "TXT")
    os.makedirs(txt_dir, exist_ok=True)

    txt_path = os.path.join(txt_dir, log_name)

    fh = logging.FileHandler(txt_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.addHandler(fh)

    logger.info(f"TXT лог хадгалалт идэвхжлээ: {txt_path}")

# ─────────────────────────────────────────────
#  RESEARCH PARAMETER OVERRIDE
# ─────────────────────────────────────────────

change_params = input(
    "\nСудалгааны параметрүүдийг өөрчлөх үү? (y/n): "
).strip().lower()

if change_params == "y":

    try:
        WINDOW_SIZE = int(input(f"WINDOW_SIZE [{WINDOW_SIZE}]: ") or WINDOW_SIZE)
        STRIDE      = int(input(f"STRIDE [{STRIDE}]: ") or STRIDE)
        EPOCHS      = int(input(f"EPOCHS [{EPOCHS}]: ") or EPOCHS)
        BATCH_SIZE  = int(input(f"BATCH_SIZE [{BATCH_SIZE}]: ") or BATCH_SIZE)
        LR          = float(input(f"LEARNING_RATE [{LR}]: ") or LR)

        logger.info("Параметрүүд шинэчлэгдлээ:")
        logger.info(f"WINDOW_SIZE = {WINDOW_SIZE}")
        logger.info(f"STRIDE      = {STRIDE}")
        logger.info(f"EPOCHS      = {EPOCHS}")
        logger.info(f"BATCH_SIZE  = {BATCH_SIZE}")
        logger.info(f"LR          = {LR}")

    except Exception as e:
        logger.error(f"Параметр өөрчлөх үед алдаа: {e}")

# ─────────────────────────────────────────────
#  ВОСПРОИЗВОДИМОСТЬ
# ─────────────────────────────────────────────

def set_seed(seed: int = SEED):
    """Фиксирует все генераторы случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Seed установлен: {seed}")

# ─────────────────────────────────────────────
#  СОХРАНЕНИЕ / ЗАГРУЗКА МАССИВОВ
# ─────────────────────────────────────────────

def save_arrays(X: np.ndarray, y: np.ndarray, prefix: str, directory: str = SPECTRO_DIR):
    """Сохраняет спектрограммы (X) и метки (y) в .npy файлы."""
    x_path = os.path.join(directory, f"{prefix}_X.npy")
    y_path = os.path.join(directory, f"{prefix}_y.npy")
    np.save(x_path, X)
    np.save(y_path, y)
    logger.info(f"Сохранено: {x_path}  (форма: {X.shape})")
    logger.info(f"Сохранено: {y_path}  (форма: {y.shape})")

def load_arrays(prefix: str, directory: str = SPECTRO_DIR):
    """Загружает спектрограммы и метки из .npy файлов."""
    x_path = os.path.join(directory, f"{prefix}_X.npy")
    y_path = os.path.join(directory, f"{prefix}_y.npy")
    X = np.load(x_path)
    y = np.load(y_path)
    logger.info(f"Загружено: X{X.shape}, y{y.shape}  из '{directory}/{prefix}_*'")
    return X, y

# ─────────────────────────────────────────────
#  РАЗБИВКА ДАННЫХ
# ─────────────────────────────────────────────

def split_data(X: np.ndarray, y: np.ndarray):
    """
    Разбивает данные на train / val / test с стратификацией.
    Возвращает: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=val_fraction,
        random_state=SEED,
        stratify=y_tmp
    )
    logger.info(
        f"Разбивка: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ─────────────────────────────────────────────
#  ВЕСА КЛАССОВ (для дисбаланса данных)
# ─────────────────────────────────────────────

def get_class_weights(y: np.ndarray) -> dict:
    """Вычисляет веса классов для несбалансированных данных."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw = dict(zip(classes.astype(int), weights))
    logger.info(f"Веса классов: {cw}")
    return cw

# ─────────────────────────────────────────────
#  ИНФОРМАЦИЯ О КОНФИГУРАЦИИ
# ─────────────────────────────────────────────

def print_config():
    logger.info("=" * 55)
    logger.info("  КОНФИГУРАЦИЯ ПРОЕКТА")
    logger.info("=" * 55)
    logger.info(f"  Режим классификации : {CLASSIFICATION_MODE}")
    logger.info(f"  Размер окна         : {WINDOW_SIZE} потоков")
    logger.info(f"  Шаг окна            : {STRIDE}")
    logger.info(f"  Кол-во признаков    : {N_FEATURES}")
    logger.info(f"  Размер спектрограммы: {IMG_H} × {IMG_W}")
    logger.info(f"  Batch size          : {BATCH_SIZE}")
    logger.info(f"  Epochs              : {EPOCHS}")
    logger.info(f"  Learning rate       : {LR}")
    logger.info("=" * 55)

# ─────────────────────────────────────────────
#  ҮР ДҮНГ TXT-Д ХАДГАЛАХ
# ─────────────────────────────────────────────

def save_log_to_txt(stage_name: str):
    """
    Тухайн үе шатны лог файлаас үр дүнг
    results/ хавтас дотор txt болгон хуулна.
    """
    answer = input(f"\nҮр дүнг TXT файл болгон хадгалах уу? (y/n): ").strip().lower()
    if answer != "y":
        logger.info("TXT хадгалахыг алгасав.")
        return

    import shutil
    from datetime import datetime

    log_src = os.path.join(BASE_DIR, "pipeline.log")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name  = f"{stage_name}_{timestamp}.txt"
    out_path  = os.path.join(RESULTS_DIR, out_name)

    if os.path.exists(log_src):
        shutil.copy2(log_src, out_path)
        logger.info(f"TXT хадгалагдлаа: {out_path}")
    else:
        logger.warning("pipeline.log олдсонгүй.")

if __name__ == "__main__":
    set_seed()
    print_config()
    logger.info("utils.py — OK")

