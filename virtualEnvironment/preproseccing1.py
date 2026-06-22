# 1 extraction features TF-IDF (VERSI RANDOM SPLIT - TANPA STRATIFIED)
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("🔄 [TAHAP 1] Memulai Preprocessing & TF-IDF...")

# --- CONFIG ---
# Gunakan Absolute Path dengan 'r' di depan agar aman di Windows
FILE_DATA = r'D:\project skripsi machine learning intoleransi\virtualEnvironment\dataset\dataYangDiPakai\data_label_3_kategori_v2 - Copy.csv' 

# Tentukan folder tujuan penyimpanan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_OUTPUT = os.path.join(BASE_DIR, 'output')
FOLDER_MODELS = os.path.join(BASE_DIR, 'models')

# 0. CEK KEAMANAN FILE SEBELUM JALAN
if not os.path.exists(FILE_DATA):
    raise FileNotFoundError(f"❌ File tidak ditemukan di jalur:\n{FILE_DATA}\nCoba pastikan nama file dan foldernya sudah persis sama!")

# Bikin folder otomatis (pakai slash '/' biar aman di semua sistem)
os.makedirs(FOLDER_OUTPUT, exist_ok=True) 
os.makedirs(FOLDER_MODELS, exist_ok=True) 

# 1. LOAD DATA
print("   - Membaca dataset...")
df = pd.read_csv(FILE_DATA, sep=';') 

# Cek keamanan kolom (Biar tidak error kalau nama kolom salah)
if 'clean_text' not in df.columns or 'label' not in df.columns:
    print(f"Daftar kolom yang ada di filemu: {df.columns.tolist()}")
    raise KeyError("❌ Kolom 'clean_text' atau 'label' tidak ada! Coba cek tulisan di atas, pastikan namanya cocok.")

# Bersihkan data kosong
df = df.dropna(subset=['clean_text', 'label']) 
print(f"   - Total data bersih yang siap diproses: {len(df)} baris")

# 2. TF-IDF (Ubah Huruf jadi Angka)
print("   - Melakukan ekstraksi fitur TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'].astype(str))
y = df['label'].astype(int) # Pastikan label berupa angka (0, 1, 2)

# 3. SIMPAN KAMUS TF-IDF
print(f"   - Menyimpan kamus TF-IDF ke '{FOLDER_MODELS}'...")
joblib.dump(vectorizer, f'{FOLDER_MODELS}/vectorizer_tfidf.pkl')

# 4. SPLIT DATA dengan RANDOM SPLIT (80% Latih, 20% Uji)
print("   - Memecah data dengan Random Split (80% Data Latih, 20% Data Uji)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4.1 TAMPILKAN HASIL DISTRIBUSI LABEL (Untuk Perbandingan dengan Stratified)
print("\n" + "="*55)
print("📊 HASIL RANDOM SPLIT (TANPA STRATIFIED)")
print("="*55)
print(f"\n📈 Total Data Awal  : {len(y)} baris")
print(f"📈 Total Data Latih : {len(y_train)} baris ({len(y_train)/len(y)*100:.1f}%)")
print(f"📈 Total Data Uji   : {len(y_test)} baris ({len(y_test)/len(y)*100:.1f}%)")

print(f"\n📊 Distribusi Label Dataset Awal:")
for label, jumlah in y.value_counts().sort_index().items():
    persen = jumlah / len(y) * 100
    print(f"   - Label {label}: {jumlah} data ({persen:.2f}%)")

print(f"\n📊 Distribusi Label Data Latih (Random Split):")
for label, jumlah in y_train.value_counts().sort_index().items():
    persen = jumlah / len(y_train) * 100
    print(f"   - Label {label}: {jumlah} data ({persen:.2f}%)")

print(f"\n📊 Distribusi Label Data Uji (Random Split):")
for label, jumlah in y_test.value_counts().sort_index().items():
    persen = jumlah / len(y_test) * 100
    print(f"   - Label {label}: {jumlah} data ({persen:.2f}%)")

print("="*55 + "\n")

# 5. SIMPAN DATA MATANG
print(f"   - Menyimpan data matang ke folder '{FOLDER_OUTPUT}'...")
joblib.dump(X_train, f'{FOLDER_OUTPUT}/X_train.pkl')
joblib.dump(X_test,  f'{FOLDER_OUTPUT}/X_test.pkl')
joblib.dump(y_train, f'{FOLDER_OUTPUT}/y_train.pkl')
joblib.dump(y_test,  f'{FOLDER_OUTPUT}/y_test.pkl')

print("✅ SELESAI TAHAP 1. Data sudah siap!")
print("👉 Silakan lanjut jalankan file '2_training.py' di terminal")