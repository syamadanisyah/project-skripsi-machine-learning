# 4 Ekstraksi Feature Importance dari SVM Linear
import joblib
import numpy as np
import pandas as pd
import os

print("🔍 [TAHAP 4] Ekstraksi Kata Penanda per Kelas (SVM Linear)...")

FOLDER_OUTPUT = 'virtualEnvironment/output'
FOLDER_MODELS = 'virtualEnvironment/models'

# Load model SVM dan vectorizer TF-IDF
svm_model = joblib.load(f'{FOLDER_MODELS}/model_svm.pkl')
vectorizer = joblib.load(f'{FOLDER_MODELS}/vectorizer_tfidf.pkl')

# Cek apakah kernel linear (kalau bukan linear, coef_ tidak tersedia)
if svm_model.kernel != 'linear':
    raise ValueError(f"❌ Kernel SVM bukan linear (saat ini: {svm_model.kernel}). Feature importance hanya bisa diekstrak dari kernel linear.")

# Ambil daftar kata dari vectorizer
daftar_kata = np.array(vectorizer.get_feature_names_out())

# Ambil koefisien SVM
# Untuk multiclass (3 kelas), shape coef_ = (n_classes, n_features) untuk linear SVC dengan ovr
# Tapi sklearn pakai ovo (one-vs-one) jadi shape = (n_classes*(n_classes-1)/2, n_features)
# Untuk 3 kelas: shape = (3, n_features) → kombinasi 0v1, 0v2, 1v2

koef = svm_model.coef_.toarray() if hasattr(svm_model.coef_, 'toarray') else svm_model.coef_
print(f"   - Shape koefisien: {koef.shape}")
print(f"   - Jumlah fitur (kata): {len(daftar_kata)}")

# Untuk OVO multiclass dengan 3 kelas:
# Baris 0: kelas 0 vs kelas 1 (positif → kelas 1, negatif → kelas 0)
# Baris 1: kelas 0 vs kelas 2 (positif → kelas 2, negatif → kelas 0)
# Baris 2: kelas 1 vs kelas 2 (positif → kelas 2, negatif → kelas 1)

label_map = {0: 'Netral', 1: 'Rasional_Negatif', 2: 'Cacimaki_Intoleransi'}

# Hitung "skor kelas" dengan menjumlahkan koefisien dari pasangan yang relevan
# Skor kelas 0 = -koef[0] (lawan kelas 1) + -koef[1] (lawan kelas 2)
# Skor kelas 1 = +koef[0] (vs kelas 0) + -koef[2] (vs kelas 2)
# Skor kelas 2 = +koef[1] (vs kelas 0) + +koef[2] (vs kelas 1)

skor_per_kelas = {
    0: -koef[0] - koef[1],  # Netral
    1:  koef[0] - koef[2],  # Rasional Negatif
    2:  koef[1] + koef[2],  # Cacimaki/Intoleransi
}

TOP_N = 20

print("\n" + "="*60)
print(f"📌 TOP {TOP_N} KATA PENANDA PER KELAS")
print("="*60)

hasil_semua = []

for kelas, skor in skor_per_kelas.items():
    nama_kelas = label_map[kelas]
    
    # Sort: ambil indeks dengan skor tertinggi
    top_idx = np.argsort(skor)[::-1][:TOP_N]
    top_kata = daftar_kata[top_idx]
    top_skor = skor[top_idx]
    
    print(f"\n🔹 Kelas {kelas} - {nama_kelas}:")
    for i, (kata, s) in enumerate(zip(top_kata, top_skor), 1):
        print(f"   {i:2d}. {kata:<25s} (skor: {s:.4f})")
        hasil_semua.append({
            'kelas': nama_kelas,
            'rank': i,
            'kata': kata,
            'skor': s
        })

# Simpan ke CSV
df_hasil = pd.DataFrame(hasil_semua)
nama_file = f'{FOLDER_OUTPUT}/feature_importance_svm.csv'
df_hasil.to_csv(nama_file, index=False, sep=';')
print(f"\n💾 Hasil tersimpan di: {nama_file}")
print("\n✅ SELESAI!")