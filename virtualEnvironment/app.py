"""
Dashboard Analisis Intoleransi Sound Horeg di Media X
Berfokus pada analisis mendalam kasus, bukan klasifikasi real-time.

Models: SVM, KNN, Ensemble (Soft Voting SVM+KNN, bobot 2:1)
Fitur: TF-IDF
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# ============================================================
# SETUP PATH
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Analisis Wacana Sound Horeg di Media X",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# KONSTANTA
# ============================================================
LABEL_MAP = {0: "Netral", 1: "Rasional Negatif", 2: "Cacimaki/Intoleransi"}
LABELS_ORDER = ["Netral", "Rasional Negatif", "Cacimaki/Intoleransi"]
COLOR_MAP = {
    "Netral": "#6B7280",            # abu-abu
    "Rasional Negatif": "#F59E0B",  # oranye
    "Cacimaki/Intoleransi": "#DC2626",  # merah
}
MODEL_COLORS = {
    "SVM": "#2563EB",
    "KNN": "#10B981",
    "Ensemble": "#8B5CF6",
}

# ============================================================
# LOADER (CACHED)
# ============================================================
@st.cache_resource
def load_models():
    svm = joblib.load(os.path.join(MODELS_DIR, "model_svm.pkl"))
    knn = joblib.load(os.path.join(MODELS_DIR, "model_knn.pkl"))
    ens = joblib.load(os.path.join(MODELS_DIR, "model_ensemble.pkl"))
    vec = joblib.load(os.path.join(MODELS_DIR, "vectorizer_tfidf.pkl"))
    return svm, knn, ens, vec


@st.cache_data
def load_dataset():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "dataset.csv"), sep=";")
    df = df.dropna(subset=["clean_text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(LABEL_MAP)
    return df


@st.cache_data
def load_kesalahan_svm():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "kesalahan_svm.csv"), sep=";")


@st.cache_data
def compute_predictions():
    """Reproduksi split skripsi: seed=42, test_size=0.2, stratify=label — identik dengan preprocessing1.py."""
    df = load_dataset()
    svm, knn, ens, vec = load_models()

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    Xv = vec.transform(X_te)

    pred_svm = svm.predict(Xv)
    pred_knn = knn.predict(Xv)
    pred_ens = ens.predict(Xv)

    out = pd.DataFrame({
        "clean_text": X_te.values,
        "label_asli": y_te.values,
        "pred_svm": pred_svm,
        "pred_knn": pred_knn,
        "pred_ens": pred_ens,
    })
    out["label_asli_name"] = out["label_asli"].map(LABEL_MAP)
    out["pred_svm_name"] = out["pred_svm"].map(LABEL_MAP)
    out["pred_knn_name"] = out["pred_knn"].map(LABEL_MAP)
    out["pred_ens_name"] = out["pred_ens"].map(LABEL_MAP)
    return out, X_tr, y_tr


@st.cache_data
def compute_feature_importance():
    """Turunkan importance per kelas dari coef_ SVM OvO.
    Sesuai Tabel 4.20 di skripsi."""
    svm, _, _, vec = load_models()
    feat = vec.get_feature_names_out()
    C = svm.coef_.toarray() if hasattr(svm.coef_, "toarray") else np.asarray(svm.coef_)

    # SVC OvO: pasangan (0,1), (0,2), (1,2)
    # Kelas 0 (Netral): sisi negatif di (0,1) dan (0,2)
    imp0 = (-C[0] + -C[1]) / 2
    # Kelas 1 (Rasional Negatif): sisi positif (0,1), sisi negatif (1,2)
    imp1 = (C[0] + -C[2]) / 2
    # Kelas 2 (Cacimaki/Intoleransi): sisi positif (0,2) dan (1,2)
    imp2 = (C[1] + C[2]) / 2

    return {
        "Netral": pd.DataFrame({"kata": feat, "skor": imp0}).sort_values("skor", ascending=False).reset_index(drop=True),
        "Rasional Negatif": pd.DataFrame({"kata": feat, "skor": imp1}).sort_values("skor", ascending=False).reset_index(drop=True),
        "Cacimaki/Intoleransi": pd.DataFrame({"kata": feat, "skor": imp2}).sort_values("skor", ascending=False).reset_index(drop=True),
    }


# ============================================================
# UTILITAS VISUALISASI
# ============================================================
def plot_confusion_matrix(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    z_text = [[str(v) for v in row] for row in cm]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=LABELS_ORDER,
        y=LABELS_ORDER,
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "white"},
        colorscale="Blues",
        showscale=True,
        hovertemplate="Asli: %{y}<br>Prediksi: %{x}<br>Jumlah: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Prediksi",
        yaxis_title="Label Asli",
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def get_metrics_table(y_true, preds_dict):
    rows = []
    for name, pred in preds_dict.items():
        acc = accuracy_score(y_true, pred)
        f1m = f1_score(y_true, pred, average="macro")
        f1w = f1_score(y_true, pred, average="weighted")
        rows.append({
            "Model": name,
            "Akurasi": f"{acc*100:.2f}%",
            "Macro F1": f"{f1m:.2f}",
            "Weighted F1": f"{f1w:.2f}",
        })
    return pd.DataFrame(rows)


def get_per_class_metrics(y_true, pred, model_name):
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2])
    return pd.DataFrame({
        "Model": [model_name]*3,
        "Kelas": LABELS_ORDER,
        "Precision": p.round(2),
        "Recall": r.round(2),
        "F1-score": f.round(2),
        "Support": s,
    })


# ============================================================
# SIDEBAR NAVIGASI
# ============================================================
st.sidebar.title("Sound Horeg Dashboard")
st.sidebar.caption("Analisis Intoleransi pada Media X")
st.sidebar.markdown("---")

PAGES = [
    "Overview Dataset",
    "Komparasi 3 Model",
    "Detail Model: SVM",
    "Detail Model: KNN",
    "Detail Model: Ensemble",
    "Analisis Kasus Sound Horeg",
    "Eksplorasi Tweet Interaktif",
]
page = st.sidebar.radio("Navigasi", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dashboard ini berfokus pada **analisis mendalam wacana** "
    "fenomena *sound horeg* di media X. Akurasi klasifikasi bukan "
    "tujuan utama — model digunakan sebagai instrumen analitis."
)

# ============================================================
# LOAD DATA SEKALI
# ============================================================
df_all = load_dataset()
preds_df, X_train_series, y_train_series = compute_predictions()

y_true = preds_df["label_asli"].values
preds_dict = {
    "SVM": preds_df["pred_svm"].values,
    "KNN": preds_df["pred_knn"].values,
    "Ensemble": preds_df["pred_ens"].values,
}


# ============================================================
# HALAMAN 1: OVERVIEW DATASET
# ============================================================
if page == "Overview Dataset":
    st.title("Overview Dataset")
    st.caption("Distribusi data, statistik dasar, dan karakteristik korpus *tweet* tentang *sound horeg*.")

    total = len(df_all)
    n_train = len(X_train_series)
    n_test = len(preds_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tweet", f"{total:,}")
    c2.metric("Data Latih", f"{n_train:,}")
    c3.metric("Data Uji", f"{n_test:,}")
    c4.metric("Jumlah Kelas", "3")

    st.markdown("### Distribusi Label")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        dist = df_all["label_name"].value_counts().reindex(LABELS_ORDER).reset_index()
        dist.columns = ["Label", "Jumlah"]
        dist["Persentase"] = (dist["Jumlah"] / dist["Jumlah"].sum() * 100).round(2)
        fig = px.bar(
            dist, x="Label", y="Jumlah", color="Label",
            color_discrete_map=COLOR_MAP, text="Jumlah",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.pie(
            dist, names="Label", values="Jumlah", color="Label",
            color_discrete_map=COLOR_MAP, hole=0.4,
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(dist, use_container_width=True, hide_index=True)

    st.markdown("### Distribusi Pada Data Uji (n=279)")
    test_dist = preds_df["label_asli_name"].value_counts().reindex(LABELS_ORDER).reset_index()
    test_dist.columns = ["Label", "Jumlah"]
    test_dist["Persentase"] = (test_dist["Jumlah"] / test_dist["Jumlah"].sum() * 100).round(2)
    st.dataframe(test_dist, use_container_width=True, hide_index=True)
    st.caption(
        "Distribusi yang relatif seimbang membuat akurasi cukup representatif "
        "sebagai indikator performa (sesuai sub-bab 4.9.1 skripsi)."
    )

    st.markdown("### Statistik Panjang Tweet")
    df_all["length"] = df_all["clean_text"].astype(str).str.split().str.len()
    stat_panjang = df_all.groupby("label_name")["length"].agg(["mean", "median", "min", "max"]).round(1)
    stat_panjang = stat_panjang.reindex(LABELS_ORDER).reset_index().rename(columns={
        "label_name": "Label", "mean": "Rata-rata", "median": "Median", "min": "Min", "max": "Max"
    })
    st.dataframe(stat_panjang, use_container_width=True, hide_index=True)

    fig_len = px.box(
        df_all, x="label_name", y="length", color="label_name",
        category_orders={"label_name": LABELS_ORDER},
        color_discrete_map=COLOR_MAP, points=False,
        labels={"length": "Jumlah kata per tweet", "label_name": "Label"}
    )
    fig_len.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_len, use_container_width=True)

    with st.expander("📄 Contoh Data Mentah"):
        st.dataframe(
            df_all[["raw_text", "clean_text", "label_name"]].head(15),
            use_container_width=True, hide_index=True
        )


# ============================================================
# HALAMAN 2: KOMPARASI 3 MODEL
# ============================================================
elif page == "Komparasi 3 Model":
    st.title("Komparasi 3 Model")
    st.caption("Perbandingan performa SVM, KNN, dan Ensemble (Soft Voting SVM+KNN dengan bobot 2:1).")

    # Metrik utama
    st.markdown("### Tabel 4.16 — Ringkasan Performa")
    metrics_df = get_metrics_table(y_true, preds_dict)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Bar chart akurasi
    col_a, col_b = st.columns([1, 1])
    with col_a:
        acc_df = pd.DataFrame({
            "Model": list(preds_dict.keys()),
            "Akurasi (%)": [accuracy_score(y_true, p)*100 for p in preds_dict.values()],
        })
        fig = px.bar(
            acc_df, x="Model", y="Akurasi (%)", color="Model",
            color_discrete_map=MODEL_COLORS, text="Akurasi (%)",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, height=400, title="Akurasi Per Model")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        f1_rows = []
        for n, p in preds_dict.items():
            f1_rows.append({"Model": n, "Tipe F1": "Macro F1", "Nilai": f1_score(y_true, p, average="macro")})
            f1_rows.append({"Model": n, "Tipe F1": "Weighted F1", "Nilai": f1_score(y_true, p, average="weighted")})
        f1_df = pd.DataFrame(f1_rows)
        fig = px.bar(
            f1_df, x="Model", y="Nilai", color="Tipe F1", barmode="group", text="Nilai",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(height=400, title="Macro F1 vs Weighted F1")
        st.plotly_chart(fig, use_container_width=True)

    # Per kelas
    st.markdown("### Tabel 4.17 — Performa Per Kelas")
    per_class_all = pd.concat([
        get_per_class_metrics(y_true, preds_dict["SVM"], "SVM"),
        get_per_class_metrics(y_true, preds_dict["KNN"], "KNN"),
        get_per_class_metrics(y_true, preds_dict["Ensemble"], "Ensemble"),
    ], ignore_index=True)
    st.dataframe(per_class_all, use_container_width=True, hide_index=True)

    metric_choice = st.selectbox(
        "Pilih metrik untuk dibandingkan visual",
        ["Precision", "Recall", "F1-score"],
        index=1
    )
    fig = px.bar(
        per_class_all, x="Kelas", y=metric_choice, color="Model",
        barmode="group", color_discrete_map=MODEL_COLORS, text=metric_choice,
        category_orders={"Kelas": LABELS_ORDER}
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)

    # 3 confusion matrix berdampingan
    st.markdown("### Confusion Matrix Ketiga Model")
    cols = st.columns(3)
    for col, (name, pred) in zip(cols, preds_dict.items()):
        with col:
            st.plotly_chart(plot_confusion_matrix(y_true, pred, title=name), use_container_width=True)

    # Insight box
    st.markdown("### 💡 Insight Komparatif")
    _ins_acc_svm = accuracy_score(y_true, preds_dict["SVM"]) * 100
    _ins_acc_knn = accuracy_score(y_true, preds_dict["KNN"]) * 100
    _ins_acc_ens = accuracy_score(y_true, preds_dict["Ensemble"]) * 100
    _ins_selisih_svm_knn = _ins_acc_svm - _ins_acc_knn
    _, _ins_r_svm, _, _ = precision_recall_fscore_support(y_true, preds_dict["SVM"], labels=[0, 1, 2])
    _, _ins_r_knn, _, _ = precision_recall_fscore_support(y_true, preds_dict["KNN"], labels=[0, 1, 2])
    _, _ins_r_ens, _, _ = precision_recall_fscore_support(y_true, preds_dict["Ensemble"], labels=[0, 1, 2])
    _ins_worst_svm = LABELS_ORDER[int(np.argmin(_ins_r_svm))]
    _ins_worst_knn = LABELS_ORDER[int(np.argmin(_ins_r_knn))]
    _ins_worst_ens = LABELS_ORDER[int(np.argmin(_ins_r_ens))]
    _ins_recall_worst_svm = float(_ins_r_svm[int(np.argmin(_ins_r_svm))])
    _ins_recall_worst_knn = float(_ins_r_knn[int(np.argmin(_ins_r_knn))])
    _ins_recall_worst_ens = float(_ins_r_ens[int(np.argmin(_ins_r_ens))])
    if _ins_acc_ens < _ins_acc_svm:
        _ins_judul = "Mengapa Ensemble justru lebih rendah dari SVM tunggal?"
        _ins_keterangan = (
            f"Temuan ini menegaskan bahwa pendekatan Ensemble **tidak selalu menghasilkan "
            f"peningkatan performa**, terutama ketika model komponen memiliki disparitas "
            f"akurasi yang besar (SVM {_ins_acc_svm:.2f}% vs KNN {_ins_acc_knn:.2f}% — "
            f"selisih {_ins_selisih_svm_knn:.2f}%)."
        )
    else:
        _ins_judul = "Ensemble vs SVM: Keunggulan Tipis dengan KNN sebagai Penentu"
        _ins_keterangan = (
            f"Ensemble ({_ins_acc_ens:.2f}%) sedikit melampaui SVM ({_ins_acc_svm:.2f}%) "
            f"dengan selisih {abs(_ins_acc_ens - _ins_acc_svm):.2f}%. Namun disparitas "
            f"antara SVM dan KNN tetap besar (selisih {_ins_selisih_svm_knn:.2f}%), "
            f"sehingga KNN masih membebani performa Ensemble secara keseluruhan."
        )
    st.info(
        f"""
**{_ins_judul}**

Ensemble menggabungkan SVM dan KNN melalui *soft voting* dengan bobot 2:1
(lebih berat ke SVM). Walaupun bobot SVM sudah dua kali lipat KNN, probabilitas
KNN yang kurang akurat tetap berkontribusi pada keputusan akhir Ensemble.

{_ins_keterangan}

Kelas yang paling sulit dideteksi per model: SVM → **{_ins_worst_svm}** (recall {_ins_recall_worst_svm:.2f}),
KNN → **{_ins_worst_knn}** (recall {_ins_recall_worst_knn:.2f}),
Ensemble → **{_ins_worst_ens}** (recall {_ins_recall_worst_ens:.2f}).
        """
    )


# ============================================================
# HALAMAN 3: DETAIL SVM
# ============================================================
elif page == "Detail Model: SVM":
    st.title("Detail Model: SVM")
    st.caption("Support Vector Machine — kernel linear, C=1. Model dengan performa terbaik pada penelitian ini.")

    acc = accuracy_score(y_true, preds_dict["SVM"])
    f1m = f1_score(y_true, preds_dict["SVM"], average="macro")
    f1w = f1_score(y_true, preds_dict["SVM"], average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi", f"{acc*100:.2f}%")
    c2.metric("Macro F1", f"{f1m:.2f}")
    c3.metric("Weighted F1", f"{f1w:.2f}")
    c4.metric("Total Salah Klasifikasi",
              int((preds_df["label_asli"] != preds_df["pred_svm"]).sum()))

    st.markdown("---")
    st.markdown("### Confusion Matrix (Gambar 4.6)")
    _cm_svm = confusion_matrix(y_true, preds_dict["SVM"], labels=[0, 1, 2])
    _svm_d = (_cm_svm[0, 0], _cm_svm[1, 1], _cm_svm[2, 2])
    _svm_caci_netral = int(_cm_svm[2, 0])
    _svm_rn_caci = int(_cm_svm[1, 2])
    _svm_caci_rn = int(_cm_svm[2, 1])
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(plot_confusion_matrix(y_true, preds_dict["SVM"], title="SVM"), use_container_width=True)
    with col_b:
        st.dataframe(get_per_class_metrics(y_true, preds_dict["SVM"], "SVM"),
                     use_container_width=True, hide_index=True)
        st.markdown(f"""
**Bacaan singkat:**
- Diagonal utama ({_svm_d[0]}, {_svm_d[1]}, {_svm_d[2]}) = prediksi benar
- Pola kesalahan terbesar: **{_svm_caci_netral} tweet Cacimaki/Intoleransi diprediksi sebagai Netral**
- Kesalahan antara Rasional Negatif ↔ Cacimaki tergolong kecil ({_svm_rn_caci} & {_svm_caci_rn} kasus) — artinya
  ketika SVM mengenali sinyal negatif, ia cukup baik membedakan jenisnya.
        """)

    # Pola kesalahan
    st.markdown("---")
    st.markdown("### Pola Kesalahan Klasifikasi (Tabel 4.18)")

    wrong = preds_df[preds_df["label_asli"] != preds_df["pred_svm"]].copy()
    wrong["pola"] = wrong["label_asli_name"] + " → " + wrong["pred_svm_name"]
    pola_counts = wrong["pola"].value_counts().reset_index()
    pola_counts.columns = ["Pola kesalahan (label asli → prediksi)", "Jumlah"]
    pola_counts["Persentase"] = (pola_counts["Jumlah"] / pola_counts["Jumlah"].sum() * 100).round(2)
    st.dataframe(pola_counts, use_container_width=True, hide_index=True)

    fig = px.bar(
        pola_counts, x="Pola kesalahan (label asli → prediksi)", y="Jumlah",
        text="Jumlah",
        color="Persentase", color_continuous_scale="Reds",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=460, xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    # Highlight 53,33%
    neg_to_neutral = wrong[
        (wrong["label_asli"].isin([1, 2])) & (wrong["pred_svm"] == 0)
    ].shape[0]
    total_wrong = len(wrong)
    pct = neg_to_neutral / total_wrong * 100
    st.error(
        f"**Temuan kunci:** {neg_to_neutral} dari {total_wrong} kesalahan "
        f"({pct:.2f}%) adalah tweet bersentimen negatif (Rasional Negatif atau "
        f"Cacimaki/Intoleransi) yang **gagal dideteksi dan justru tergolong "
        f"ke kelas Netral**. Batas leksikal antara wacana negatif dan netral "
        f"pada fenomena *sound horeg* memang tidak selalu tegas."
    )

    # Daftar lengkap kesalahan SVM dari CSV
    st.markdown("---")
    st.markdown("### 📋 Daftar Tweet Salah Klasifikasi SVM")
    st.caption("Seluruh kasus kesalahan klasifikasi SVM — dapat difilter per pola untuk analisis kualitatif.")

    df_kesalahan = load_kesalahan_svm()

    pola_unik = ["Semua"] + sorted(df_kesalahan["pola_kesalahan"].unique().tolist())
    col_fil1, col_fil2 = st.columns([1, 3])
    with col_fil1:
        pola_pilih = st.selectbox("Filter pola kesalahan", pola_unik, key="svm_kes_filter")

    if pola_pilih != "Semua":
        df_kes_view = df_kesalahan[df_kesalahan["pola_kesalahan"] == pola_pilih].reset_index(drop=True)
    else:
        df_kes_view = df_kesalahan.copy()

    st.caption(f"Menampilkan **{len(df_kes_view)}** dari {len(df_kesalahan)} kasus kesalahan SVM.")

    st.dataframe(
        df_kes_view.rename(columns={
            "clean_text": "Tweet (setelah preprocessing)",
            "label_asli": "Label Asli",
            "label_prediksi": "Prediksi SVM",
            "pola_kesalahan": "Pola Kesalahan",
        }),
        use_container_width=True,
        hide_index=True,
        height=460,
    )

    csv_kes = df_kes_view.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        "💾 Download tabel kesalahan (CSV)",
        data=csv_kes,
        file_name="kesalahan_svm_filtered.csv",
        mime="text/csv",
        key="dl_kes_svm",
    )

    # 4 karakteristik
    st.markdown("---")
    st.markdown("### Empat Karakteristik Penyebab Kesulitan Klasifikasi")
    st.caption("Analisis kualitatif dari sub-bab 4.9.1 skripsi.")

    karakteristik = [
        ("1️⃣ Kritik rasional yang disisipi kata kasar",
         "Kritik valid tetapi memuat kata seperti *anjing*, *anjir* sebagai ekspresi spontan. "
         "Kata-kata utama yang membawa muatan keluhan tidak terasa cukup negatif secara leksikal, "
         "sementara kata kasar tersebut lebih sering muncul pada kelas cacimaki — sehingga sinyalnya membingungkan. "
         "Model memilih kelas Netral sebagai keputusan *aman*."),
        ("2️⃣ Cacimaki dengan struktur kalimat panjang dan asosiatif",
         "Cacimaki disampaikan lewat rangkaian asosiasi (tokoh politik, ijazah, kutukan). "
         "Bobot fitur kata cacimaki tersebar di antara kata netral lain, sehingga model menyimpulkan tweet sebagai netral."),
        ("3️⃣ Cacimaki yang dikemas dalam idiom atau metafora",
         "Misal *otak kebanyakan denger sound horeg makanya gapunya pikir* (metafora dehumanisasi) atau "
         "doa kutukan *moga2 cpt laknat tuhan*. Tanpa makian eksplisit, model gagal mengenali muatan intoleransi."),
        ("4️⃣ Tweet sangat pendek dengan konteks ambigu",
         "Kasus ekstrem seperti *punya sound horeg* — bahkan manusia sulit menentukan sentimennya tanpa konteks. "
         "Model justru memprediksi Rasional Negatif karena bobot kata *sound horeg* condong ke kelas negatif."),
    ]
    for title, desc in karakteristik:
        with st.expander(title, expanded=False):
            st.write(desc)

    # Kata kunci diskriminatif
    st.markdown("---")
    st.markdown("### Kata Kunci Diskriminatif (Tabel 4.20)")
    st.caption(
        "Top-20 kata dengan koefisien tertinggi dari SVM kernel linear, "
        "merepresentasikan *penanda leksikal* yang paling kuat membedakan tiap kelas."
    )

    fi = compute_feature_importance()
    n_top = st.slider("Jumlah kata teratas", 5, 30, 20, key="fi_slider")

    tabs = st.tabs(LABELS_ORDER)
    for tab, label in zip(tabs, LABELS_ORDER):
        with tab:
            top_df = fi[label].head(n_top).copy()
            top_df["rank"] = range(1, len(top_df) + 1)
            fig = px.bar(
                top_df.sort_values("skor"),
                x="skor", y="kata", orientation="h",
                color="skor", color_continuous_scale="Viridis",
                title=f"Top-{n_top} kata diskriminatif — {label}",
                height=max(380, n_top * 22),
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Interpretasi Kata Kunci")
    st.info(
        """
- **Netral**: didominasi kata bermuatan negatif (*anjir*, *bakar*, *bodoh*, *gila*, *mati*).
  Kata kasar pada media sosial Indonesia tidak selalu menunjukkan intoleransi — sering kali
  sekadar ekspresi spontan. Kata seperti *kaca*, *kuping*, *rumah*, *jawa*, *budaya*
  menunjukkan **dimensi fisik** dan **dimensi budaya** hadir di seluruh wacana.

- **Rasional Negatif**: muncul *neng* (hasil *stemming* dari *seneng*) dan *cinta* sebagai
  penanda — pola **sarkasme dan ironi** yang khas pada kritik di media X
  (*"seneng banget ya warga sini sama sound horeg"*).

- **Cacimaki/Intoleransi**: kata *prabowo* dan *karnaval* mengindikasikan **dimensi politik**
  pada wacana intoleransi. Bukan berarti tokoh tersebut intoleran, melainkan menunjukkan
  bahwa cacimaki *sound horeg* sering disertai konteks politik tertentu.
        """
    )


# ============================================================
# HALAMAN 4: DETAIL KNN
# ============================================================
elif page == "Detail Model: KNN":
    st.title("Detail Model: KNN")
    st.caption("K-Nearest Neighbors — n_neighbors=13, metric=cosine, weights=uniform.")

    acc = accuracy_score(y_true, preds_dict["KNN"])
    f1m = f1_score(y_true, preds_dict["KNN"], average="macro")
    f1w = f1_score(y_true, preds_dict["KNN"], average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi", f"{acc*100:.2f}%")
    c2.metric("Macro F1", f"{f1m:.2f}")
    c3.metric("Weighted F1", f"{f1w:.2f}")
    c4.metric("Total Salah Klasifikasi",
              int((preds_df["label_asli"] != preds_df["pred_knn"]).sum()))

    st.markdown("---")
    st.markdown("### Confusion Matrix (Gambar 4.7)")
    _cm_knn = confusion_matrix(y_true, preds_dict["KNN"], labels=[0, 1, 2])
    _knn_d = (_cm_knn[0, 0], _cm_knn[1, 1], _cm_knn[2, 2])
    _knn_rn_netral = int(_cm_knn[1, 0])
    _knn_caci_netral = int(_cm_knn[2, 0])
    _knn_non_netral_netral = _knn_rn_netral + _knn_caci_netral
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(plot_confusion_matrix(y_true, preds_dict["KNN"], title="KNN"), use_container_width=True)
    with col_b:
        st.dataframe(get_per_class_metrics(y_true, preds_dict["KNN"], "KNN"),
                     use_container_width=True, hide_index=True)
        st.markdown(f"""
**Bacaan singkat:**
- Diagonal utama ({_knn_d[0]}, {_knn_d[1]}, {_knn_d[2]}) — semua lebih rendah dari SVM
- KNN memiliki kesalahan lebih besar di hampir semua sel dibanding SVM
- Total **{_knn_non_netral_netral} tweet non-netral diprediksi sebagai netral** ({_knn_rn_netral} Rasional Negatif + {_knn_caci_netral} Cacimaki/Intoleransi)
        """)

    # Pola kesalahan KNN
    st.markdown("---")
    st.markdown("### Pola Kesalahan KNN")

    wrong = preds_df[preds_df["label_asli"] != preds_df["pred_knn"]].copy()
    wrong["pola"] = wrong["label_asli_name"] + " → " + wrong["pred_knn_name"]
    pola_counts = wrong["pola"].value_counts().reset_index()
    pola_counts.columns = ["Pola kesalahan (label asli → prediksi)", "Jumlah"]
    pola_counts["Persentase"] = (pola_counts["Jumlah"] / pola_counts["Jumlah"].sum() * 100).round(2)
    st.dataframe(pola_counts, use_container_width=True, hide_index=True)

    fig = px.bar(
        pola_counts, x="Pola kesalahan (label asli → prediksi)", y="Jumlah",
        text="Jumlah",
        color="Persentase", color_continuous_scale="Greens",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=460, xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Mengapa KNN 'Menarik' Prediksi ke Kelas Mayoritas?")
    st.warning(
        f"""
KNN bekerja dengan mengukur kedekatan jarak antar dokumen menggunakan **cosine similarity**.
Karakteristik kerjanya menyebabkan beberapa kesulitan pada data TF-IDF yang berdimensi tinggi:

1. **Data teks bersifat *sparse*** — banyak fitur bernilai nol, sehingga pengukuran jarak
   antar dokumen pendek menjadi kurang stabil. Dua tweet yang berbeda kelas dapat memiliki
   *cosine similarity* tinggi hanya karena berbagi beberapa kata umum.

2. **Tidak ada batas keputusan eksplisit** — KNN tidak membangun *hyperplane* seperti SVM;
   prediksinya bergantung penuh pada kemiripan permukaan antar tweet.

3. **Tarikan ke kelas mayoritas** — ketika tetangga terdekat berasal dari berbagai kelas,
   KNN cenderung memilih kelas paling sering muncul (Netral di sini), terutama pada
   tweet pendek yang minim fitur khas.

Inilah mengapa **{_knn_non_netral_netral} tweet non-netral salah diprediksi sebagai netral** —
pola "tarikan ke netral" lebih kuat pada KNN dibanding SVM.
        """
    )

    # Perbandingan kasus KNN vs SVM
    st.markdown("---")
    st.markdown("### Kasus Dimana KNN Salah tapi SVM Benar")
    knn_wrong_svm_right = preds_df[
        (preds_df["pred_knn"] != preds_df["label_asli"]) &
        (preds_df["pred_svm"] == preds_df["label_asli"])
    ][["clean_text", "label_asli_name", "pred_svm_name", "pred_knn_name"]].rename(columns={
        "clean_text": "Tweet",
        "label_asli_name": "Label Asli",
        "pred_svm_name": "Prediksi SVM",
        "pred_knn_name": "Prediksi KNN",
    })
    st.caption(f"Ditemukan {len(knn_wrong_svm_right)} kasus di mana KNN gagal tetapi SVM benar.")
    st.dataframe(knn_wrong_svm_right, use_container_width=True, hide_index=True, height=400)


# ============================================================
# HALAMAN 5: DETAIL ENSEMBLE
# ============================================================
elif page == "Detail Model: Ensemble":
    st.title("Detail Model: Ensemble")
    st.caption("Soft Voting Classifier — kombinasi SVM dan KNN dengan bobot 2:1.")

    acc = accuracy_score(y_true, preds_dict["Ensemble"])
    f1m = f1_score(y_true, preds_dict["Ensemble"], average="macro")
    f1w = f1_score(y_true, preds_dict["Ensemble"], average="weighted")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi", f"{acc*100:.2f}%")
    c2.metric("Macro F1", f"{f1m:.2f}")
    c3.metric("Weighted F1", f"{f1w:.2f}")
    c4.metric("Total Salah Klasifikasi",
              int((preds_df["label_asli"] != preds_df["pred_ens"]).sum()))

    st.markdown("---")
    st.markdown("### Confusion Matrix (Gambar 4.8)")
    _cm_ens = confusion_matrix(y_true, preds_dict["Ensemble"], labels=[0, 1, 2])
    _ens_d = (_cm_ens[0, 0], _cm_ens[1, 1], _cm_ens[2, 2])
    _ens_rn_netral = int(_cm_ens[1, 0])
    _ens_caci_netral = int(_cm_ens[2, 0])
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(plot_confusion_matrix(y_true, preds_dict["Ensemble"], title="Ensemble"), use_container_width=True)
    with col_b:
        st.dataframe(get_per_class_metrics(y_true, preds_dict["Ensemble"], "Ensemble"),
                     use_container_width=True, hide_index=True)
        st.markdown(f"""
**Bacaan singkat:**
- Diagonal utama ({_ens_d[0]}, {_ens_d[1]}, {_ens_d[2]}) — lebih baik dari KNN tetapi tidak lebih baik dari SVM tunggal
- {_ens_rn_netral} kasus Rasional Negatif dan {_ens_caci_netral} kasus Cacimaki/Intoleransi diprediksi sebagai Netral
- Pengaruh prediksi KNN terlihat dari pola kesalahan yang sedikit lebih banyak dibanding SVM
        """)

    # Pola kesalahan Ensemble
    st.markdown("---")
    st.markdown("### Pola Kesalahan Ensemble")

    wrong = preds_df[preds_df["label_asli"] != preds_df["pred_ens"]].copy()
    wrong["pola"] = wrong["label_asli_name"] + " → " + wrong["pred_ens_name"]
    pola_counts = wrong["pola"].value_counts().reset_index()
    pola_counts.columns = ["Pola kesalahan (label asli → prediksi)", "Jumlah"]
    pola_counts["Persentase"] = (pola_counts["Jumlah"] / pola_counts["Jumlah"].sum() * 100).round(2)
    st.dataframe(pola_counts, use_container_width=True, hide_index=True)

    fig = px.bar(
        pola_counts, x="Pola kesalahan (label asli → prediksi)", y="Jumlah",
        text="Jumlah",
        color="Persentase", color_continuous_scale="Purples",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=460, xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    # Efek soft voting 2:1
    st.markdown("---")
    _ens_acc_e = accuracy_score(y_true, preds_dict["Ensemble"]) * 100
    _ens_acc_s = accuracy_score(y_true, preds_dict["SVM"]) * 100
    _ens_judul_voting = (
        "Efek Soft Voting 2:1 — Mengapa Ensemble Lebih Rendah dari SVM?"
        if _ens_acc_e < _ens_acc_s
        else "Efek Soft Voting 2:1 — Ensemble vs SVM: Analisis Keputusan"
    )
    st.markdown(f"### {_ens_judul_voting}")

    # Hitung agreement
    agree_all = ((preds_df["pred_svm"] == preds_df["pred_knn"]) &
                 (preds_df["pred_svm"] == preds_df["pred_ens"])).sum()
    svm_eq_ens = (preds_df["pred_svm"] == preds_df["pred_ens"]).sum()
    knn_eq_ens = (preds_df["pred_knn"] == preds_df["pred_ens"]).sum()
    svm_knn_disagree = (preds_df["pred_svm"] != preds_df["pred_knn"]).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("SVM ≠ KNN", svm_knn_disagree,
              help="Jumlah kasus di mana SVM dan KNN tidak sepakat. Di sinilah voting menentukan.")
    c2.metric("Ensemble = SVM", svm_eq_ens,
              help="Berapa kali Ensemble mengikuti SVM.")
    c3.metric("Ensemble = KNN", knn_eq_ens,
              help="Berapa kali Ensemble mengikuti KNN.")

    # Kasus dimana SVM benar tapi Ensemble salah (karena KNN menarik)
    svm_right_ens_wrong = preds_df[
        (preds_df["pred_svm"] == preds_df["label_asli"]) &
        (preds_df["pred_ens"] != preds_df["label_asli"])
    ]

    st.error(
        f"**{len(svm_right_ens_wrong)} kasus** di mana SVM sudah benar, tetapi probabilitas "
        f"KNN yang kurang akurat menarik keputusan Ensemble menjadi salah. Inilah konsekuensi "
        f"langsung dari disparitas akurasi yang besar antar model komponen — bobot 2:1 belum "
        f"cukup untuk meredam pengaruh KNN sepenuhnya."
    )

    with st.expander(f"📄 Lihat {len(svm_right_ens_wrong)} kasus SVM benar tapi Ensemble salah"):
        st.dataframe(
            svm_right_ens_wrong[["clean_text", "label_asli_name", "pred_svm_name", "pred_knn_name", "pred_ens_name"]].rename(columns={
                "clean_text": "Tweet",
                "label_asli_name": "Label Asli",
                "pred_svm_name": "SVM",
                "pred_knn_name": "KNN",
                "pred_ens_name": "Ensemble",
            }),
            use_container_width=True, hide_index=True, height=400
        )


# ============================================================
# HALAMAN 6: ANALISIS KASUS SOUND HOREG (BAB 4.10)
# ============================================================
elif page == "Analisis Kasus Sound Horeg":
    st.title("Analisis Kasus Sound Horeg")
    st.caption("Telaah mendalam fenomena *sound horeg* melalui tiga sudut pandang yang saling berhubungan (Bab 4.10).")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📜 Tipologi Wacana (4.10.1)",
        "🔊 Dimensi Fisik & Kesehatan (4.10.2)",
        "🎭 Dimensi Budaya & Tradisi (4.10.3)",
        "🧩 Sintesis (4.10.4)",
    ])

    # ---- TAB 1: Tipologi ----
    with tab1:
        st.markdown("### Tipologi Wacana Sound Horeg di Media X")
        st.write(
            "Ujaran negatif terhadap *sound horeg* di media X memiliki **sasaran yang beragam** "
            "(fenomena, pelaku, kelompok, suku, individu) dan **bentuk penyampaian yang tidak selalu eksplisit**. "
            "Sebagian besar muncul sebagai sindiran halus, pertanyaan, atau candaan bermuatan negatif — "
            "sehingga batas antara *rasional negatif* dan *cacimaki/intoleransi* menjadi tipis."
        )

        st.markdown("#### Tabel 4.21 — Contoh Ujaran Representasi Wacana Negatif")
        tipologi_data = [
            (1, "@ramaArsy1 HAPUS SOUND HOREG!!! MERESAHKAN GAK GUNAAA CUMA NGERUSAK DAN GANGGU ORANGGGG!!!!!!! btw yg ga terima sini maju lu anj.", "Rasional negatif – penolakan eksplisit"),
            (2, "TOLOL BEGO ANJING BINATANG HARAM GADA OTAK YANG NYIPTAIN SOUND HOREG BABI.", "Cacimaki/intoleransi – serangan terhadap individu"),
            (3, "@V3g3L Kebanggaan apa yg didapat dari pemilik sound horeg merusak rumah dan fasilitas umum ?.", "Rasional negatif – dampak dan pertanyaan"),
            (4, "Sound horeg. Kerasukan kuda lumping. Mulyono. Jawa destruktif.", "Cacimaki/intoleransi – serangan individu dan kelompok"),
            (5, "Toa adzan disebut terlalu keras tapi sound horeg dimaklumi wkwkwkw agak lain si amang.", "Negatif rasional – berupa candaan yang fakta"),
            (6, "@ramaArsy1 Norak bener moga2 cpt dilaknat Tuhan yg masang sound horeg.", "Cacimaki/intoleransi – ujaran halus/sarkas berbentuk doa"),
            (7, "polusi suara anj gw nggk tau apa gunanya sound horeg ini selain bikin kuping budeg.", "Rasional negatif – berupa dampak fisik"),
            (8, "sudah saatnya mengapresiasi sound horeg semoga semakin menggelegar sampek bisa bikin rubuh rumah para penikmat dan pemilik sound horeg tsb.", "Rasional negatif – ujaran halus/sarkas positif"),
            (9, "ntar kalo meninggoy kuburannya diisi sound horeg dipepet ke kuping sampe kupingnya pecah.", "Cacimaki/intoleransi – berupa candaan kematian"),
            (10, "@vanc1Bozz @ramaArsy1 Berarti kalau ada yang lempar molly ke sound horeg netizen tambah seneng.", "Cacimaki/intoleransi – berbentuk sarkas candaan ancaman"),
            (11, "@ramaArsy1 Semakin zalim ke sesama pecinta sound horeg semakin senang (pridavan sektor timur).", "Cacimaki/intoleransi – sarkas membawa nama kelompok"),
            (12, "@CutSarina5 Sound horeg ini adalah budaya jatim..jangan sampai diredam bahkan diklaim negara tetangga..harus dijaga dilestarikan..paham ya.", "Cacimaki/intoleransi – sarkas membawa nama daerah dan budaya"),
            (13, "@yoyouzhii Serame sound horeg. Nt pasti sdh pada dijajah sound horeg yang kesenian pada hilang.", "Rasional negatif – berbentuk pernyataan tentang kesenian"),
            (14, "@anonjawa Ga semua orang jawa suka sama sound horeg ya. Gausa bawa2 suku.", "Rasional negatif – membawa nama kelompok/daerah"),
            (15, "@KakMugi ini baru enak diliat wkwkkw banyak2in lah bakar2 sound Horeg biar punah.", "Cacimaki/intoleransi – sarkas candaan halus ancaman kelompok"),
            (16, "Baguslah terbakar sekalian orang yg mengadakan Sound Horeg ini kan gara2 Jokowi bisnis sound system sepi order jaman SBY banyak panggilan.", "Cacimaki/intoleransi – sarkas candaan tentang rasa senang terhadap peristiwa dan tokoh politik"),
            (17, "@tekarok007 masyarakat tidak butuh gagasan ga butuh belajar. KITA BUTUH MITOS KITA BUTUH DONGENG KITA BUTUH BANSOS KITA BUTUH SOUND HOREG !!!.", "Rasional negatif – sarkas seruan"),
        ]
        tipologi_df = pd.DataFrame(tipologi_data, columns=["No", "Ujaran", "Arah Tipe Wacana"])

        # Filter
        all_kategori = ["Semua"] + sorted({
            "Rasional negatif" if "Rasional" in s else "Cacimaki/intoleransi"
            for s in tipologi_df["Arah Tipe Wacana"]
        })
        pilih = st.selectbox("Filter kategori utama", all_kategori, key="tipologi_filter")
        if pilih != "Semua":
            view = tipologi_df[tipologi_df["Arah Tipe Wacana"].str.contains(pilih, case=False)]
        else:
            view = tipologi_df
        st.dataframe(view, use_container_width=True, hide_index=True, height=480)

        st.info(
            "**Keberagaman sasaran dan bentuk penyampaian** inilah salah satu sumber kesulitan "
            "model klasifikasi dalam membedakan ujaran *rasional negatif* dan *cacimaki/intoleransi*."
        )

    # ---- TAB 2: Dimensi Fisik ----
    with tab2:
        st.markdown("### Dimensi Gangguan Fisik dan Kesehatan")
        st.write(
            "Sebagian besar ujaran pada Tabel 4.21 secara eksplisit menyinggung dampak fisik yang "
            "dirasakan masyarakat sekitar. Dua bentuk perusakan yang muncul:"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🏠 Perusakan Properti")
            st.markdown("""
- **Disengaja**: pembongkaran atau pemindahan struktur tertentu untuk memperlancar arak-arakan *sound system*
- **Tidak disengaja**: gelombang suara berintensitas tinggi yang menghasilkan getaran merusak
- Contoh ujaran: *"merusak rumah dan fasilitas umum"* (Tweet #3),
  *"bisa bikin rubuh rumah para penikmat"* (Tweet #8)
            """)
        with col_b:
            st.markdown("#### 👂 Gangguan Pendengaran")
            st.markdown("""
- **Permanen**: penurunan kemampuan mendengar yang tidak dapat dipulihkan
- **Sementara**: penurunan fungsi pendengaran yang masih dapat pulih
- Contoh ujaran: *"polusi suara"*, *"bikin kuping budeg"* (Tweet #7)

*Sumber: Hamdiyati & Wibawanti (2024)*
            """)

        st.markdown("---")
        st.markdown("#### Kata-kata Penanda Dimensi Fisik di Korpus")
        kata_fisik = ["kaca", "kuping", "rumah", "pecah", "berisik", "polusi", "rusak", "speaker", "soundnya"]
        fi = compute_feature_importance()
        rows = []
        for kata in kata_fisik:
            for label in LABELS_ORDER:
                df_label = fi[label]
                row = df_label[df_label["kata"] == kata]
                if not row.empty:
                    rows.append({
                        "Kata": kata,
                        "Kelas": label,
                        "Skor Koefisien": row.iloc[0]["skor"],
                        "Rank": df_label[df_label["kata"] == kata].index[0] + 1,
                    })
        kata_fisik_df = pd.DataFrame(rows)
        if not kata_fisik_df.empty:
            fig = px.bar(
                kata_fisik_df, x="Kata", y="Skor Koefisien", color="Kelas",
                barmode="group", color_discrete_map=COLOR_MAP,
                category_orders={"Kelas": LABELS_ORDER},
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        st.success(
            "Ujaran-ujaran negatif yang menyinggung dimensi fisik dan kesehatan **bukan ujaran "
            "tanpa dasar**. Ujaran tersebut mencerminkan keluhan berlandaskan empiris — baik "
            "kerusakan properti maupun risiko kesehatan."
        )

    # ---- TAB 3: Dimensi Budaya ----
    with tab3:
        st.markdown("### Dimensi Budaya dan Tradisi: Pertentangan Nilai")
        st.write(
            "Fenomena *sound horeg* tidak lepas dari dimensi budaya dan tradisi yang membentuk "
            "interaksi sosial. Pertentangan terjadi antara dua kelompok pemaknaan:"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🎉 Pelaku Sound Horeg (Remaja)")
            st.markdown("""
- *Sound horeg* dimaknai sebagai **sarana ekspresi diri** dan kebebasan berekspresi
- Bagian dari proses pencarian jati diri pada kalangan remaja
- Dentuman suara dan modifikasi *speaker* dianggap **medium performatif** yang memberi kebanggaan sosial
- Pelampiasan dari rutinitas atau masalah lingkungan sekolah

*Sumber: Auliana dkk. (2025)*
            """)
        with col_b:
            st.markdown("#### 🕌 Masyarakat Umum & Tokoh Keagamaan")
            st.markdown("""
- **Masyarakat terdampak**: *sound horeg* merusak ketertiban sosial
- **Tokoh keagamaan (pesantren tradisional, MUI)**: simbol **dekadensi moral** dan hilangnya adab di ruang publik
- Dipandang sebagai pelanggaran norma sosial

*Sumber: Mohammad Fikri dkk. (2025)*
            """)

        st.markdown("---")
        st.markdown("#### Kata-kata Penanda Dimensi Budaya & Politik di Korpus")
        kata_budaya = ["jawa", "budaya", "karnaval", "prabowo", "pariwisata", "tradisi", "santri"]
        fi = compute_feature_importance()
        rows = []
        for kata in kata_budaya:
            for label in LABELS_ORDER:
                df_label = fi[label]
                row = df_label[df_label["kata"] == kata]
                if not row.empty:
                    rows.append({
                        "Kata": kata,
                        "Kelas": label,
                        "Skor Koefisien": row.iloc[0]["skor"],
                    })
        kata_budaya_df = pd.DataFrame(rows)
        if not kata_budaya_df.empty:
            fig = px.bar(
                kata_budaya_df, x="Kata", y="Skor Koefisien", color="Kelas",
                barmode="group", color_discrete_map=COLOR_MAP,
                category_orders={"Kelas": LABELS_ORDER},
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Fenomena *sound horeg* memperlihatkan dinamika kompleks antara **ekspresi budaya "
            "populer anak muda** dan **struktur kuasa nilai** dari masyarakat dan tokoh "
            "keagamaan — sebuah gambaran kebudayaan yang mengekspresikan identitas pribadi "
            "dalam lingkungan sosial."
        )

    # ---- TAB 4: Sintesis ----
    with tab4:
        st.markdown("### Sintesis: Sound Horeg sebagai Cerminan Ketegangan Sosial Digital")

        st.write(
            "Hasil klasifikasi ketiga model (SVM, KNN, Ensemble) dan analisis wacana saling "
            "memvalidasi satu argumen utama:"
        )

        _sint_wrong_svm = preds_df[preds_df["label_asli"] != preds_df["pred_svm"]]
        _sint_neg_netral = _sint_wrong_svm[
            (_sint_wrong_svm["label_asli"].isin([1, 2])) & (_sint_wrong_svm["pred_svm"] == 0)
        ].shape[0]
        _sint_total_wrong = len(_sint_wrong_svm)
        _sint_pct = _sint_neg_netral / _sint_total_wrong * 100 if _sint_total_wrong > 0 else 0
        st.markdown(
            f"""
> **Kegagalan model mendeteksi {_sint_pct:.2f}% ujaran negatif bukan kelemahan teknis semata,
> melainkan cerminan dari kompleksitas wacana sound horeg di media X.**
            """
        )

        st.markdown("---")
        st.markdown("#### Empat Karakteristik Cara Penyampaian Ketidakpuasan")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**1. Kritik rasional disisipi kata kasar**
*"tidak bisa tidur gara2 sound horeg pawai hari santri lgian hari santri malah nyetel dj an si anjing"*

**2. Struktur kalimat panjang dan asosiatif**
*"mending bunuh sekarang gede jadi jawir suka sound horeg ijazah palsu ngaku lulus ugm jadi presiden amit amit jabang bayi setan"*
            """)
        with col_b:
            st.markdown("""
**3. Cacimaki dikemas idiom/metafora**
*"otak kebanyakan denger sound horeg makanya gapunya pikir"*

**4. Doa kutukan**
*"norak bener moga2 cpt laknat tuhan masang sound horeg"*
            """)

        st.markdown("---")
        st.markdown("#### Validasi melalui Kata Kunci Diskriminatif")
        st.markdown(
            """
- **Dimensi fisik & budaya** hadir di seluruh kelas wacana (kata *kaca*, *kuping*, *rumah*,
  *jawa*, *budaya* sebagai penanda kuat kelas **Netral** — bukti bahwa topik ini muncul di
  semua nada penyampaian, hanya gaya berbeda).
- **Strategi sarkasme** divalidasi oleh kemunculan kata *neng* (dari *seneng*) dan *cinta*
  sebagai penanda kelas **Rasional Negatif** — pengguna ingin menyalurkan ketidakpuasan
  tanpa terbaca sebagai intoleran.
- **Dimensi politik** muncul lewat kata *prabowo* dan *karnaval* pada kelas
  **Cacimaki/Intoleransi** — wacana *sound horeg* meluas dari sekadar kebisingan menjadi
  kritik terhadap figur dan peristiwa publik.
            """
        )

        st.markdown("---")
        st.success(
            """
**Kesimpulan akhir:**
Wacana *sound horeg* di media X merupakan **titik temu berbagai bentuk ketegangan sosial
yang lebih luas** — fisik, budaya, dan politik. Ketiga model klasifikasi yang berbasis
representasi TF-IDF (fitur leksikal) mengenali ujaran negatif eksplisit dengan baik,
tetapi kesulitan menangkap strategi linguistik halus seperti sarkasme, idiom, dan
doa kutukan. **Performa ketiga model adalah cerminan langsung dari kondisi ruang
sosial digital di Indonesia**, bukan kelemahan teknis semata.
            """
        )


# ============================================================
# HALAMAN 7: EKSPLORASI TWEET INTERAKTIF
# ============================================================
elif page == "Eksplorasi Tweet Interaktif":
    st.title("Eksplorasi Tweet Interaktif")
    st.caption(
        "Telusuri data uji (279 tweet) dengan filter agreement/disagreement antar model — "
        "ini cara paling produktif menemukan **kasus sulit** yang menarik untuk dianalisis."
    )

    explore_df = preds_df.copy()
    explore_df["sepakat_3"] = (
        (explore_df["pred_svm"] == explore_df["pred_knn"]) &
        (explore_df["pred_svm"] == explore_df["pred_ens"])
    )
    explore_df["svm_benar"] = explore_df["pred_svm"] == explore_df["label_asli"]
    explore_df["knn_benar"] = explore_df["pred_knn"] == explore_df["label_asli"]
    explore_df["ens_benar"] = explore_df["pred_ens"] == explore_df["label_asli"]
    explore_df["semua_benar"] = (explore_df["svm_benar"] & explore_df["knn_benar"] & explore_df["ens_benar"])
    explore_df["semua_salah"] = (~explore_df["svm_benar"] & ~explore_df["knn_benar"] & ~explore_df["ens_benar"])

    st.markdown("### Filter")
    col1, col2, col3 = st.columns(3)

    with col1:
        label_filter = st.multiselect(
            "Label Asli", LABELS_ORDER, default=LABELS_ORDER, key="exp_label"
        )

    with col2:
        agreement_filter = st.selectbox(
            "Mode Filter Kesepakatan Model",
            [
                "Semua",
                "Ketiganya sepakat",
                "Ketiganya tidak sepakat",
                "Hanya SVM yang benar (KNN & Ensemble salah)",
                "Hanya KNN yang benar (SVM & Ensemble salah)",
                "Hanya Ensemble yang benar (SVM & KNN salah)",
                "Semua model salah",
                "Semua model benar",
            ],
            key="exp_agreement",
        )

    with col3:
        search = st.text_input("Cari kata kunci dalam tweet", "", key="exp_search").strip().lower()

    # Apply filters
    view = explore_df[explore_df["label_asli_name"].isin(label_filter)]

    if agreement_filter == "Ketiganya sepakat":
        view = view[view["sepakat_3"]]
    elif agreement_filter == "Ketiganya tidak sepakat":
        # Semua 3 prediksi berbeda
        view = view[
            (view["pred_svm"] != view["pred_knn"]) &
            (view["pred_svm"] != view["pred_ens"]) &
            (view["pred_knn"] != view["pred_ens"])
        ]
    elif agreement_filter == "Hanya SVM yang benar (KNN & Ensemble salah)":
        view = view[view["svm_benar"] & ~view["knn_benar"] & ~view["ens_benar"]]
    elif agreement_filter == "Hanya KNN yang benar (SVM & Ensemble salah)":
        view = view[~view["svm_benar"] & view["knn_benar"] & ~view["ens_benar"]]
    elif agreement_filter == "Hanya Ensemble yang benar (SVM & KNN salah)":
        view = view[~view["svm_benar"] & ~view["knn_benar"] & view["ens_benar"]]
    elif agreement_filter == "Semua model salah":
        view = view[view["semua_salah"]]
    elif agreement_filter == "Semua model benar":
        view = view[view["semua_benar"]]

    if search:
        view = view[view["clean_text"].str.lower().str.contains(search, na=False)]

    st.markdown(f"### Hasil: {len(view)} tweet")

    if len(view) == 0:
        st.warning("Tidak ada tweet yang cocok dengan filter ini.")
    else:
        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total tweet", len(view))
        c2.metric("SVM benar", int(view["svm_benar"].sum()),
                  delta=f"{view['svm_benar'].mean()*100:.1f}%", delta_color="off")
        c3.metric("KNN benar", int(view["knn_benar"].sum()),
                  delta=f"{view['knn_benar'].mean()*100:.1f}%", delta_color="off")
        c4.metric("Ensemble benar", int(view["ens_benar"].sum()),
                  delta=f"{view['ens_benar'].mean()*100:.1f}%", delta_color="off")

        # Tampilkan hasil
        show_cols = view[[
            "clean_text", "label_asli_name",
            "pred_svm_name", "pred_knn_name", "pred_ens_name"
        ]].rename(columns={
            "clean_text": "Tweet",
            "label_asli_name": "Label Asli",
            "pred_svm_name": "SVM",
            "pred_knn_name": "KNN",
            "pred_ens_name": "Ensemble",
        })
        st.dataframe(show_cols, use_container_width=True, hide_index=True, height=520)

        # Download
        csv = show_cols.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download hasil filter (CSV)",
            data=csv,
            file_name="hasil_eksplorasi_tweet.csv",
            mime="text/csv",
        )

    # Distribusi tweet sulit
    st.markdown("---")
    st.markdown("### Distribusi Tingkat Kesulitan Tweet")
    n_benar = explore_df[["svm_benar", "knn_benar", "ens_benar"]].sum(axis=1)
    diff_df = pd.DataFrame({
        "Kategori": [
            "Semua model benar (mudah)",
            "2 dari 3 benar",
            "Hanya 1 dari 3 benar",
            "Semua model salah (sulit)",
        ],
        "Jumlah": [
            int((n_benar == 3).sum()),
            int((n_benar == 2).sum()),
            int((n_benar == 1).sum()),
            int((n_benar == 0).sum()),
        ],
    })
    fig = px.bar(
        diff_df, x="Kategori", y="Jumlah", color="Kategori",
        text="Jumlah",
        color_discrete_sequence=["#10B981", "#84CC16", "#F59E0B", "#DC2626"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Tweet di kategori **\"Semua model salah\"** adalah kandidat paling kaya untuk "
        "analisis kualitatif kasus sulit — gunakan filter di atas untuk menelusuri."
    )
