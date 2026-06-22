# Distribusi prediksi SVM pada keseluruhan dataset
import joblib
import pandas as pd
import numpy as np

FOLDER_OUTPUT = 'virtualEnvironment/output'
FOLDER_MODELS = 'virtualEnvironment/models'

# Load model dan vectorizer
svm_model = joblib.load(f'{FOLDER_MODELS}/model_svm.pkl')
vectorizer = joblib.load(f'{FOLDER_MODELS}/vectorizer_tfidf.pkl')

# Load dataset asli
FILE_DATA = r'D:\project skripsi machine learning intoleransi\virtualEnvironment\dataset\dataYangDiPakai\data_label_3_kategori_v2 - Copy.csv'
df = pd.read_csv(FILE_DATA, sep=';')
df = df.dropna(subset=['clean_text', 'label']).reset_index(drop=True)

print(f"Total tweet di dataset: {len(df)}")

# Transform pakai vectorizer yang sama
X_all = vectorizer.transform(df['clean_text'].astype(str))

# Prediksi semua tweet
y_pred_all = svm_model.predict(X_all)

# Hitung distribusi prediksi
label_map = {0: 'Netral', 1: 'Rasional Negatif', 2: 'Cacimaki/Intoleransi'}
distribusi_prediksi = pd.Series(y_pred_all).value_counts().sort_index()

print("\n📊 DISTRIBUSI PREDIKSI SVM PADA KESELURUHAN DATA:")
for label, jumlah in distribusi_prediksi.items():
    persen = jumlah / len(y_pred_all) * 100
    print(f"   - {label_map[label]}: {jumlah} tweet ({persen:.2f}%)")

# Bandingkan dengan distribusi label asli
print("\n📊 DISTRIBUSI LABEL ASLI:")
distribusi_asli = df['label'].astype(int).value_counts().sort_index()
for label, jumlah in distribusi_asli.items():
    persen = jumlah / len(df) * 100
    print(f"   - {label_map[label]}: {jumlah} tweet ({persen:.2f}%)")

# Simpan dataframe lengkap dengan prediksi (untuk eksplorasi lanjut)
df['prediksi_svm'] = [label_map[x] for x in y_pred_all]
df['label_asli_nama'] = [label_map[x] for x in df['label'].astype(int)]
df.to_csv(f'{FOLDER_OUTPUT}/dataset_dengan_prediksi.csv', index=False, sep=';')
print(f"\n💾 Dataset dengan prediksi tersimpan: {FOLDER_OUTPUT}/dataset_dengan_prediksi.csv")