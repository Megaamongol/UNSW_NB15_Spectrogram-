"""
run_pipeline.py — Полный запуск пайплайна одной командой

Запуск: python run_pipeline.py
  --step all       # Все этапы (по умолчанию)
  --step preprocess
  --step spectrograms
  --step train
  --step evaluate
"""

import argparse
import subprocess
import sys
import os

STEPS = {
    "preprocess"  : "src/preprocess.py",
    "spectrograms": "src/spectrogram_gen.py",
    "train"       : "src/train.py",
    "evaluate"    : "src/evaluate.py",
}

def run_step(script_path: str):
    print(f"\n{'='*55}")
    print(f"  Запуск: {script_path}")
    print(f"{'='*55}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print(f"\n[ОШИБКА] {script_path} завершился с кодом {result.returncode}")
        sys.exit(result.returncode)
    print(f"  [OK] {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Пайплайн: обнаружение сетевых атак по спектрограммам (UNSW-NB15)"
    )
    parser.add_argument(
        "--step",
        choices=["all"] + list(STEPS.keys()),
        default="all",
        help="Этап для выполнения (default: all)"
    )
    args = parser.parse_args()

    if args.step == "all":
        for name, path in STEPS.items():
            run_step(path)
    else:
        run_step(STEPS[args.step])

    print("\n" + "="*55)
    print("  ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО")
    print("  Результаты: папка results/")
    print("="*55)


if __name__ == "__main__":
    main()
