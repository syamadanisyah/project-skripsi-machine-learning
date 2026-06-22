# 3 evaluasi hasil model
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

print("📊 [TAHAP 3] Evaluasi Hasil Model...")

# --- PENYESUAIAN FOLDER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_OUTPUT = os.path.join(BASE_DIR, 'output')
FOLDER_MODELS = os.path.join(BASE_DIR, 'models')
FOLDER_IMAGES = os.path.join(BASE_DIR, 'images')

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