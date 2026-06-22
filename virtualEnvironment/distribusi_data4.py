# 3.2 Menampilkan Distribusi Data
# Menampilkan perbandingan jumlah data dan tampilan grafik perbandingan data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Pastikan folder images ada untuk menyimpan hasil
os.makedirs('images', exist_ok=True)

# ==========================================
# 1. LOAD DATA
# ==========================================
nama_file = r'd:\project skripsi machine learning intoleransi\virtualEnvironment\dataset\dataYangDiPakai\data_label_revisi_goblog.csv'

print(f"📂 Membaca file: {nama_file}...")
df = pd.read_csv(nama_file, sep=';') 

# ==========================================
# 2. HITUNG JUMLAH LABEL
# ==========================================
# Kita pastikan kolom 'label' ada
if 'label' not in df.columns:
    raise ValueError("❌ Kolom 'label' tidak ditemukan di dataset! Coba cek separatornya (sep=';' atau sep=',')")

jumlah_label = df['label'].value_counts().sort_index()

print("\n📊 STATISTIK JUMLAH DATA:")
print("-" * 30)
label_names = {0: "Netral (0)", 1: "Kritik (1)", 2: "Hujatan (2)"}
for lbl, count in jumlah_label.items():
    print(f"   {label_names.get(lbl, lbl)}: {count} data")
print("-" * 30)
print(f"   TOTAL: {len(df)} data")

# ==========================================
# 3. BUAT GRAFIK (VISUALISASI)
# ==========================================
plt.figure(figsize=(8, 6)) # Ukuran gambar (Lebar, Tinggi)

# Bikin Bar Chart warna-warni (tambah hue=... agar tidak muncul warning di versi Seaborn terbaru)
ax = sns.barplot(x=jumlah_label.index, y=jumlah_label.values, hue=jumlah_label.index, palette='viridis', legend=False)

# Hiasan Grafik
plt.title('Perbandingan Jumlah Data per Label', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Kategori Label', fontsize=12, fontweight='bold')
plt.ylabel('Jumlah Data', fontsize=12, fontweight='bold')

# Pastikan urutan label sesuai dengan 0, 1, 2
urutan_label = sorted(jumlah_label.index.tolist())
plt.xticks(ticks=range(len(urutan_label)), labels=['0\n(Netral)', '1\n(Kritik)', '2\n(Hujatan)'])
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Tampilkan Angka di Atas Batang
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize=14, fontweight='bold', color='black')

# Simpan Gambar di dalam folder 'images'
nama_gambar = 'virtualEnvironment/images/grafik_distribusi_data.png'
plt.savefig(nama_gambar, dpi=300, bbox_inches='tight')
print(f"\n🖼️ Grafik berhasil disimpan: {nama_gambar}")

# Tampilkan gambar di layar
plt.show()