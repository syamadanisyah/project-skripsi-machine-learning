# 3 evaluasi hasil model
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

print("📊 [TAHAP 3] Evaluasi Hasil Model...")

# --- PENYESUAIAN FOLDER ---
FOLDER_OUTPUT = 'virtualEnvironment/output'
FOLDER_MODELS = 'virtualEnvironment/models'
FOLDER_IMAGES = 'virtualEnvironment/images'

# 0. CEK KEAMANAN
file_xtest = f'{FOLDER_OUTPUT}/X_test.pkl'
file_svm = f'{FOLDER_MODELS}/model_svm.pkl'

if not os.path.exists(file_xtest):
    raise FileNotFoundError(f"❌ File '{file_xtest}' tidak ditemukan! Pastikan Tahap 1 sudah dijalankan.")
if not os.path.exists(file_svm):
    raise FileNotFoundError(f"❌ File '{file_svm}' tidak ditemukan! Pastikan Tahap 2 sudah selesai 100%.")

# Bikin folder images otomatis di dalam virtualEnvironment
os.makedirs(FOLDER_IMAGES, exist_ok=True)

# 1. Ambil Data Uji
print("   - Memuat data uji...")
X_test = joblib.load(file_xtest)
y_test = joblib.load(f'{FOLDER_OUTPUT}/y_test.pkl')

# 2. Daftar Model
print("   - Memuat model-model AI...")
daftar_model = {
    "SVM": joblib.load(file_svm),
    "KNN": joblib.load(f'{FOLDER_MODELS}/model_knn.pkl'),
    "Ensemble": joblib.load(f'{FOLDER_MODELS}/model_ensemble.pkl')
}

# 3. Loop Evaluasi
for nama, model in daftar_model.items():
    print(f"\n==========================================")
    print(f"--- Evaluasi Model: {nama} ---")
    print(f"==========================================")

    # Lakukan Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung Akurasi
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Akurasi {nama}: {acc*100:.2f}%\n")
    
    # Laporan Lengkap (Precision, Recall, F1-Score)
    print("📋 Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    # Bikin Grafik Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                annot_kws={"size": 14}) # Angka di dalam kotak diperbesar
    
    # Hiasan Grafik (Lebih rapi untuk masuk ke buku Skripsi)
    plt.title(f'Confusion Matrix - {nama}\n(Akurasi: {acc*100:.2f}%)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Prediksi Mesin', fontsize=12, fontweight='bold')
    plt.ylabel('Label Asli', fontsize=12, fontweight='bold')
    
    # Tambahan Keterangan Sumbu X dan Y biar dosen mudah baca
    plt.xticks(ticks=[0.5, 1.5, 2.5], labels=['0 (Netral)', '1 (rasional negatif)', '2 (cacimaki/intoleransi)'])
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=['0 (Netral)', '1 (rasional negatif)', '2 (cacimaki/intoleransi)'], rotation=0)

    # Simpan Gambar dengan resolusi tinggi (dpi=300)
    nama_file = f'{FOLDER_IMAGES}/cm_{nama}.png'
    plt.savefig(nama_file, bbox_inches='tight', dpi=300)
    print(f"🖼️ Grafik Confusion Matrix tersimpan: {nama_file}")
    
    # Tutup plot supaya tidak numpuk di memori
    plt.close()

print("\n🎉 SEMUA TAHAPAN SELESAI!")
print(f"Cek folder '{FOLDER_IMAGES}' untuk melihat gambar Confusion Matrix-nya ya.")

# 3 evaluasi hasil model
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

print("📊 [TAHAP 3] Evaluasi Hasil Model...")

# --- PENYESUAIAN FOLDER ---
FOLDER_OUTPUT = 'virtualEnvironment/output'
FOLDER_MODELS = 'virtualEnvironment/models'
FOLDER_IMAGES = 'virtualEnvironment/images'

# 0. CEK KEAMANAN
file_xtest = f'{FOLDER_OUTPUT}/X_test.pkl'
file_svm = f'{FOLDER_MODELS}/model_svm.pkl'

if not os.path.exists(file_xtest):
    raise FileNotFoundError(f"❌ File '{file_xtest}' tidak ditemukan! Pastikan Tahap 1 sudah dijalankan.")
if not os.path.exists(file_svm):
    raise FileNotFoundError(f"❌ File '{file_svm}' tidak ditemukan! Pastikan Tahap 2 sudah selesai 100%.")

# Bikin folder images otomatis di dalam virtualEnvironment
os.makedirs(FOLDER_IMAGES, exist_ok=True)

# 1. Ambil Data Uji
print("   - Memuat data uji...")
X_test = joblib.load(file_xtest)
y_test = joblib.load(f'{FOLDER_OUTPUT}/y_test.pkl')

# 2. Daftar Model
print("   - Memuat model-model AI...")
daftar_model = {
    "SVM": joblib.load(file_svm),
    "KNN": joblib.load(f'{FOLDER_MODELS}/model_knn.pkl'),
    "Ensemble": joblib.load(f'{FOLDER_MODELS}/model_ensemble.pkl')
}

# 3. Loop Evaluasi
for nama, model in daftar_model.items():
    print(f"\n==========================================")
    print(f"--- Evaluasi Model: {nama} ---")
    print(f"==========================================")

    # Lakukan Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung Akurasi
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Akurasi {nama}: {acc*100:.2f}%\n")
    
    # Laporan Lengkap (Precision, Recall, F1-Score)
    print("📋 Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    # Bikin Grafik Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                annot_kws={"size": 14}) # Angka di dalam kotak diperbesar
    
    # Hiasan Grafik (Lebih rapi untuk masuk ke buku Skripsi)
    plt.title(f'Confusion Matrix - {nama}\n(Akurasi: {acc*100:.2f}%)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Prediksi Mesin', fontsize=12, fontweight='bold')
    plt.ylabel('Label Asli', fontsize=12, fontweight='bold')
    
    # Tambahan Keterangan Sumbu X dan Y biar dosen mudah baca
    plt.xticks(ticks=[0.5, 1.5, 2.5], labels=['0 (Netral)', '1 (rasional negatif)', '2 (cacimaki/intoleransi)'])
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=['0 (Netral)', '1 (rasional negatif)', '2 (cacimaki/intoleransi)'], rotation=0)

    # Simpan Gambar dengan resolusi tinggi (dpi=300)
    nama_file = f'{FOLDER_IMAGES}/cm_{nama}.png'
    plt.savefig(nama_file, bbox_inches='tight', dpi=300)
    print(f"🖼️ Grafik Confusion Matrix tersimpan: {nama_file}")
    
    # Tutup plot supaya tidak numpuk di memori
    plt.close()

print("\n🎉 SEMUA TAHAPAN SELESAI!")
print(f"Cek folder '{FOLDER_IMAGES}' untuk melihat gambar Confusion Matrix-nya ya.")

# ============================================
# TAMBAHAN: Ekstraksi Pola Kesalahan SVM
# ============================================
print("\n" + "="*60)
print("📋 EKSTRAKSI POLA KESALAHAN KLASIFIKASI (SVM)")
print("="*60)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load ulang data asli untuk dapat teks tweet
FILE_DATA = r'D:\project skripsi machine learning intoleransi\virtualEnvironment\dataset\dataYangDiPakai\data_label_3_kategori_v2 - Copy.csv'
df_asli = pd.read_csv(FILE_DATA, sep=';')
df_asli = df_asli.dropna(subset=['clean_text', 'label']).reset_index(drop=True)

# Replikasi split yang sama dengan Tahap 1 (random_state=42)
indeks_semua = np.arange(len(df_asli))
_, indeks_test = train_test_split(indeks_semua, test_size=0.2, random_state=42)

# Prediksi pakai SVM
y_pred_svm = daftar_model['SVM'].predict(X_test)
y_test_arr = np.array(y_test)

# Filter yang salah
salah_mask = y_pred_svm != y_test_arr
indeks_salah = indeks_test[salah_mask]

# Bikin dataframe kesalahan
label_map = {0: 'Netral', 1: 'Rasional Negatif', 2: 'Cacimaki/Intoleransi'}
df_salah = df_asli.loc[indeks_salah].copy()
df_salah['label_asli'] = [label_map[x] for x in y_test_arr[salah_mask]]
df_salah['label_prediksi'] = [label_map[x] for x in y_pred_svm[salah_mask]]
df_salah['pola_kesalahan'] = df_salah['label_asli'] + ' → ' + df_salah['label_prediksi']

# Simpan ke CSV
nama_file_salah = f'{FOLDER_OUTPUT}/kesalahan_svm.csv'
df_salah[['clean_text', 'label_asli', 'label_prediksi', 'pola_kesalahan']].to_csv(
    nama_file_salah, index=False, sep=';'
)

# Statistik pola kesalahan
print(f"\n📊 Total kesalahan SVM: {len(df_salah)} dari {len(y_test_arr)} data uji")
print(f"\n📊 Distribusi Pola Kesalahan:")
print(df_salah['pola_kesalahan'].value_counts().to_string())
print(f"\n💾 Detail tersimpan di: {nama_file_salah}")