# 2 pelatihan model SVM, KNN dan ensemble
import joblib
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

print("🏋️ [TAHAP 2] Training: DATA ASLI (Mode TURBO Aktif 🚀)...")

# --- PENYESUAIAN FOLDER ---
FOLDER_OUTPUT = 'virtualEnvironment/output'
FOLDER_MODELS = 'virtualEnvironment/models'

# 0. CEK KEAMANAN: Pastikan folder dan file Tahap 1 sudah ada
file_xtrain = f'{FOLDER_OUTPUT}/X_train.pkl'
if not os.path.exists(file_xtrain):
    raise FileNotFoundError(f"❌ File '{file_xtrain}' tidak ditemukan! Pastikan kamu sudah menjalankan '1_preprocessing.py' terlebih dahulu.")

# Jaga-jaga kalau folder models belum ada, otomatis dibikin
os.makedirs(FOLDER_MODELS, exist_ok=True) 

# 1. AMBIL DATA DARI TAHAP 1
print("   - Memuat data latih...")
X_train = joblib.load(f'{FOLDER_OUTPUT}/X_train.pkl')
y_train = joblib.load(f'{FOLDER_OUTPUT}/y_train.pkl')

print(f"   - Jumlah Data Latih Asli: {len(y_train)} baris")
print(f"   - Komposisi Label: {y_train.value_counts().to_dict()}")

# ---------------------------------------------------------
# 2. LATIH SVM (MODEL UTAMA)
# ---------------------------------------------------------
print("\n🚀 Melatih SVM (Mencari Settingan Terbaik)...")
print("   (Menggunakan seluruh inti CPU laptop...)")

param_svm = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear'],
    'gamma': ['scale']
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42), 
    param_svm, 
    cv=5, 
    scoring='f1_macro', 
    verbose=1, 
    n_jobs=-1
      )

svm_grid.fit(X_train, y_train) 

best_svm = svm_grid.best_estimator_
joblib.dump(best_svm, f'{FOLDER_MODELS}/model_svm.pkl')

# ⭐ TAMBAHAN: Print akurasi validasi & parameter terbaik SVM
svm_acc_val = svm_grid.best_score_ * 100
print(f"   ✅ SVM Selesai (F1-Macro Validasi: {svm_acc_val:.2f}%)")
print(f"   📌 Parameter Terbaik SVM: {svm_grid.best_params_}")

# ---------------------------------------------------------
# 3. LATIH KNN (METRIC COSINE)
# ---------------------------------------------------------
print("\n🚀 Melatih KNN (Wajib Cosine)...")

param_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 25], 
    'metric': ['cosine'],
    'weights': ['uniform', 'distance']
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(), 
    param_knn, 
    cv=5,
    scoring='f1_macro', 
    verbose=1, 
    n_jobs=-1
    )
knn_grid.fit(X_train, y_train)

best_knn = knn_grid.best_estimator_
joblib.dump(best_knn, f'{FOLDER_MODELS}/model_knn.pkl')

# ⭐ TAMBAHAN: Print akurasi validasi & parameter terbaik KNN
knn_acc_val = knn_grid.best_score_ * 100
print(f"   ✅ KNN Selesai (Best K: {knn_grid.best_params_['n_neighbors']})")
print(f"   📌 Parameter Terbaik KNN: {knn_grid.best_params_}")
print(f"   📌 F1-Macro Validasi KNN: {knn_acc_val:.2f}%")

# ---------------------------------------------------------
# 4. LATIH ENSEMBLE (SVM + KNN)
# ---------------------------------------------------------
print("\n🚀 Melatih ENSEMBLE (Voting SVM + KNN)...")

# Gabungkan dua model terbaik
ensemble_model = VotingClassifier(
    estimators=[
        ('svm', best_svm), 
        ('knn', best_knn)
    ],
    voting='soft',
    weights=[2, 1] # SVM kita beri bobot suara lebih tinggi
)

ensemble_model.fit(X_train, y_train)
joblib.dump(ensemble_model, f'{FOLDER_MODELS}/model_ensemble.pkl')
print("   ✅ Ensemble Selesai.")

# ⭐ TAMBAHAN: Hitung akurasi validasi Ensemble pakai cross_val_score
print("   📌 Menghitung Akurasi Validasi Ensemble (3-fold CV)...")
print("      (Sabar ya, ini agak lama karena ensemble = SVM + KNN x 3 fold)")
ensemble_scores = cross_val_score(ensemble_model, X_train, y_train, cv=3, n_jobs=-1)
ensemble_acc_val = ensemble_scores.mean() * 100
print(f"   📌 Akurasi Validasi Ensemble: {ensemble_acc_val:.2f}%")

# ---------------------------------------------------------
# 5. RINGKASAN AKHIR
# ---------------------------------------------------------
print("\n" + "="*60)
print("📊 RINGKASAN HASIL TRAINING")
print("="*60)
print(f"\n🔹 SVM")
print(f"   Parameter : {svm_grid.best_params_}")
print(f"   Akurasi Validasi : {svm_acc_val:.2f}%")

print(f"\n🔹 KNN")
print(f"   Parameter : {knn_grid.best_params_}")
print(f"   Akurasi Validasi : {knn_acc_val:.2f}%")

print(f"\n🔹 Ensemble (SVM + KNN)")
print(f"   Voting    : soft, weights=[2, 1]")
print(f"   Akurasi Validasi : {ensemble_acc_val:.2f}%")

print("\n" + "="*60)
print("🎉 TRAINING DATA MURNI SELESAI!")
print("👉 Silakan jalankan '3_evaluasi.py' untuk melihat hasil akhirnya.")
print("="*60)