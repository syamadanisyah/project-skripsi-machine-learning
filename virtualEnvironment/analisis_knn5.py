# 4 mencari nilai K
import joblib
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

print("🔬 [EKSPERIMEN] Membandingkan 3 Metode Mencari Nilai K Terbaik...")

# --- PENYESUAIAN FOLDER ---
FOLDER_OUTPUT = 'virtualEnvironment/output'
FOLDER_IMAGES = 'virtualEnvironment/images'

# 0. CEK KEAMANAN
file_xtrain = f'{FOLDER_OUTPUT}/X_train.pkl'
if not os.path.exists(file_xtrain):
    raise FileNotFoundError(f"❌ File '{file_xtrain}' tidak ditemukan! Pastikan Tahap 1 sudah dijalankan.")

# Pastikan folder images ada di dalam virtualEnvironment
os.makedirs(FOLDER_IMAGES, exist_ok=True)

# 1. LOAD DATA
X_train = joblib.load(file_xtrain)
y_train = joblib.load(f'{FOLDER_OUTPUT}/y_train.pkl')
jumlah_data = X_train.shape[0]

print(f"   - Jumlah Data Latih: {jumlah_data} baris")
print("-" * 50)

results = [] # Untuk menyimpan hasil perbandingan

# =========================================================
# METODE 1: AKAR KUADRAT (Square Root Rule)
# Rumus: K = Akar(Total Data)
# =========================================================
print("1️⃣ Menguji Metode Akar Kuadrat...")
k_sqrt = int(math.sqrt(jumlah_data))

# Aturan: K harus ganjil biar gak seri (draw)
if k_sqrt % 2 == 0:
    k_sqrt += 1

# Uji Akurasinya (Pakai n_jobs=-1 biar ngebut)
knn_sq = KNeighborsClassifier(n_neighbors=k_sqrt, metric='cosine')
scores_sq = cross_val_score(knn_sq, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
acc_sq = scores_sq.mean() * 100

print(f"   -> Hasil: K={k_sqrt}, Akurasi={acc_sq:.2f}%")
results.append({'Metode': 'Akar Kuadrat', 'K': k_sqrt, 'Akurasi': acc_sq})

# =========================================================
# METODE 2: ELBOW METHOD (Metode Siku)
# Coba manual dari 1 sampai 40, lalu cari error terkecil
# =========================================================
print("\n2️⃣ Menguji Metode Elbow (Looping 1-40)...")
print("   (Tunggu sebentar, sedang menghitung manual...)")
error_rates = []
acc_rates = []
k_range = range(1, 41, 2) # Coba angka ganjil: 1, 3, 5, ... 39

best_k_elbow = 0
best_acc_elbow = 0

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    # Pakai n_jobs=-1 di sini juga biar loopingnya gak kelamaan
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    acc = scores.mean()

    # Simpan data buat grafik
    acc_rates.append(acc)
    error_rates.append(1 - acc) # Error = 100% - Akurasi

    # Cek apakah ini rekor terbaik?
    if acc > best_acc_elbow:
        best_acc_elbow = acc
        best_k_elbow = k

print(f"   -> Hasil Terbaik di Range Ini: K={best_k_elbow}, Akurasi={best_acc_elbow*100:.2f}%")
results.append({'Metode': 'Elbow (Manual)', 'K': best_k_elbow, 'Akurasi': best_acc_elbow*100})

# Bikin Grafik Elbow
plt.figure(figsize=(10, 6))
plt.plot(k_range, error_rates, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=8)
plt.title('Grafik Elbow (Mencari Error Terkecil)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Nilai K', fontsize=12, fontweight='bold')
plt.ylabel('Tingkat Error', fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

# Simpan ke folder images
nama_gambar_elbow = f'{FOLDER_IMAGES}/grafik_elbow_knn.png'
plt.savefig(nama_gambar_elbow, dpi=300, bbox_inches='tight')
print(f"   🖼️ Grafik Elbow tersimpan: {nama_gambar_elbow}")
plt.close() # Tutup grafik biar memori lega

# =========================================================
# METODE 3: GRID SEARCH CV (Validasi Silang)
# Ini metode paling 'Sultan' dan Valid
# =========================================================
print("\n3️⃣ Menguji Metode Grid Search CV (Otomatis)...")
param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 15, 19, 21, 25, 29]}
grid = GridSearchCV(KNeighborsClassifier(metric='cosine'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

k_grid = grid.best_params_['n_neighbors']
acc_grid = grid.best_score_ * 100

print(f"   -> Hasil: K={k_grid}, Akurasi={acc_grid:.2f}%")
results.append({'Metode': 'Grid Search CV', 'K': k_grid, 'Akurasi': acc_grid})

# =========================================================
# KESIMPULAN AKHIR
# =========================================================
print("\n" + "="*50)
print("🏆 TABEL PERBANDINGAN METODE PENENTUAN K")
print("="*50)
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
print("-" * 50)

# Cari pemenang
best_method = df_res.loc[df_res['Akurasi'].idxmax()]
print(f"✅ REKOMENDASI: Gunakan K = {best_method['K']}")
print(f"   (Berdasarkan metode {best_method['Metode']} dengan akurasi tertinggi)")