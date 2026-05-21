# src/check_columns.py
import pandas as pd
from pathlib import Path

# Төслийн үндсэн зам – шаардлагатай бол өөрчилнө үү
BASE_DIR = Path(__file__).resolve().parent.parent   # UNSW_NB15_Spectrogram хавтас

# CSV файлын зам (data/raw/ дотор байгаа)
csv_path = BASE_DIR / "data" / "raw" / "UNSW-NB15_merged.csv"

# Хэрэв файл өөр газар байвал замыг гараар зааж болно:
# csv_path = "C:/Users/megaa/Desktop/UNSW_NB15_Spectrogram/data/raw/UNSW-NB15_merged.csv"

if not csv_path.exists():
    print(f"Файл олдсонгүй: {csv_path}")
    print("Замаа шалгаад дахин оролдоно уу.")
    exit(1)

# CSV-г унших (том файл тул эхний хэсгийг л үзүүлнэ)
df = pd.read_csv(csv_path)

# Баганы нэрс
print("=" * 50)
print(f"Файл: {csv_path.name}")
print(f"Нийт мөр: {df.shape[0]}, Багана: {df.shape[1]}")
print("=" * 50)
print("\nБаганы нэрс:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:3d}. {col}")

# Баганы төрөл
print("\nБаганы төрөл (dtype):")
print(df.dtypes)

# Эхний 5 мөр
print("\nЭхний 5 мөр:")
print(df.head().to_string())