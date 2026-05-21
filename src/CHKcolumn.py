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

# CSV-г унших
df = pd.read_csv(csv_path)

# Мэдээллийг цуглуулах (дэлгэцэнд хэвлэх болон файлд хадгалахад ашиглана)
info_lines = []                # Мөр бүрийг хадгалах жагсаалт

# Толгой мэдээлэл
info_lines.append("=" * 50)
info_lines.append(f"Файл: {csv_path.name}")
info_lines.append(f"Нийт мөр: {df.shape[0]}, Багана: {df.shape[1]}")
info_lines.append("=" * 50)

# Баганы нэрс
info_lines.append("\nБаганы нэрс:")
for i, col in enumerate(df.columns, 1):
    info_lines.append(f"  {i:3d}. {col}")

# Баганы төрөл
info_lines.append("\nБаганы төрөл (dtype):")
info_lines.append(df.dtypes.to_string())

# Эхний 5 мөр
info_lines.append("\nЭхний 5 мөр:")
# to_string-г индентацтай хадгалах
info_lines.append(df.head().to_string())

# Дэлгэцэнд бүгдийг хэвлэх
print("\n".join(info_lines))

# ===== ШИНЭ НЭМЭГДСЭН ХЭСЭГ – txt файл үүсгэх эсэхийг асууж, хадгалах =====
save_txt = input("\nҮр дүнг txt файлаар хадгалах уу? (Yes/No): ").strip().lower()
if save_txt in ["yes", "y"]:
    # results хавтсыг (байхгүй бол үүсгэх) бэлдэнэ
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # txt файлын зам
    txt_path = results_dir / "UNSW_NB15_columns.txt"
    
    # Файлд бичих
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines))
    
    print(f"[✓] Үр дүн хадгалагдлаа: {txt_path}")
else:
    print("[!] txt файл үүсгэхгүйгээр дууслаа.")