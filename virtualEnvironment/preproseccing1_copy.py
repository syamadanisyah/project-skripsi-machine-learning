# 1 extraction features TF-IDF (VERSI MODIFIKASI - STRATIFIED + BIGRAM)
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("🔄 [TAHAP 1] Memulai Preprocessing & TF-IDF...")

# --- CONFIG ---
FILE_DATA = r'D:\project skripsi machine learning intoleransi\virtualEnvironment\dataset\dataYangDiPakai\data_label_3_kategori_v2 - Copy.csv' 

FOLDER_OUTPUT = 'virtualEnvironment/output'
FOLDER_MODELS = 'virtualEnvironment/models'

# 0. CEK KEAMANAN FILE
if not os.path.exists(FILE_DATA):
    raise FileNotFoundError(f"❌ File tidak ditemukan di jalur:\n{FILE_DATA}")

os.makedirs(FOLDER_OUTPUT, exist_ok=True) 
os.makedirs(FOLDER_MODELS, exist_ok=True) 

# 1. LOAD DATA
print("   - Membaca dataset...")
df = pd.read_csv(FILE_DATA, sep=';') 

if 'clean_text' not in df.columns or 'label' not in df.columns:
    print(f"Daftar kolom: {df.columns.tolist()}")
    raise KeyError("❌ Kolom 'clean_text' atau 'label' tidak ada!")

df = df.dropna(subset=['clean_text', 'label']) 
print(f"   - Total data bersih: {len(df)} baris")

# 2. TF-IDF (MODIFIKASI: Tambah bigram, min_df, max_df)
print("   - Melakukan ekstraksi fitur TF-IDF (Unigram + Bigram)...")
vectorizer = TfidfVectorizer(
    max_features=10000,      # naik dari 5000
    ngram_range=(1, 2),      # tambah bigram
    min_df=2,                # buang kata yang muncul cuma 1x
    max_df=0.95              # buang kata yang muncul di >95% dokumen
)
X = vectorizer.fit_transform(df['clean_text'].astype(str))
y = df['label'].astype(int)

print(f"   - Jumlah fitur TF-IDF aktual: {X.shape[1]}")

# 3. SIMPAN VECTORIZER
print(f"   - Menyimpan vectorizer ke '{FOLDER_MODELS}'...")
joblib.dump(vectorizer, f'{FOLDER_MODELS}/vectorizer_tfidf.pkl')

# 4. SPLIT DATA dengan STRATIFIED (80% Latih, 20% Uji)
print("   - Memecah data dengan Stratified Split (menjaga distribusi label)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y                # ← PERUBAHAN UTAMA: stratify
)

# 4.1 TAMPILKAN HASIL DISTRIBUSI LABEL
print("\n" + "="*55)
print("📊 HASIL STRATIFIED SPLIT")
print("="*55)
print(f"\n📈 Total Data Awal  : {len(y)} baris")
print(f"📈 Total Data Latih : {len(y_train)} baris ({len(y_train)/len(y)*100:.1f}%)")
print(f"📈 Total Data Uji   : {len(y_test)} baris ({len(y_test)/len(y)*100:.1f}%)")

print(f"\n📊 Distribusi Label Dataset Awal:")
for label, jumlah in y.value_counts().sort_index().items():
    persen = jumlah / len(y) * 100
    print(f"   - Label {label}: {jumlah} data ({persen:.2f}%)")

print(f"\n📊 Distribusi Label Data Latih:")
for label, jumlah in y_train.value_counts().sort_index().items():
    persen = jumlah / len(y_train) * 100
    print(f"   - Label {label}: {jumlah} data ({persen:.2f}%)")

print(f"\n📊 Distribusi Label Data Uji:")
for label, jumlah in y_test.value_counts().sort_index().items():
    persen = jumlah / len(y_test) * 100
    print(f"   - Label {label}: {jumlah} data ({persen:.2f}%)")

print("="*55 + "\n")

# 5. SIMPAN DATA MATANG
print(f"   - Menyimpan data matang ke '{FOLDER_OUTPUT}'...")
joblib.dump(X_train, f'{FOLDER_OUTPUT}/X_train.pkl')
joblib.dump(X_test,  f'{FOLDER_OUTPUT}/X_test.pkl')
joblib.dump(y_train, f'{FOLDER_OUTPUT}/y_train.pkl')
joblib.dump(y_test,  f'{FOLDER_OUTPUT}/y_test.pkl')

print("✅ SELESAI TAHAP 1. Lanjut jalankan 'training2.py'")