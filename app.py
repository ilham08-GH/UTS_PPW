import streamlit as st
import os
import re
import joblib
import numpy as np
import trafilatura
# Inference berbasis dictionary vektor (tanpa kompilasi gensim C/C++)
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Set konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Berita - Skip-gram & Naive Bayes",
    page_icon="📰",
    layout="wide"
)

# Custom CSS untuk tampilan modern & elegan
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }
    .engine-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-sg {
        background-color: #E0E7FF;
        color: #3730A3;
    }
    .badge-nb {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .status-sport {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 8px;
        font-size: 1.4rem;
        display: inline-block;
    }
    .status-finance {
        background-color: #DBEAFE;
        color: #1E40AF;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 8px;
        font-size: 1.4rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# FUNGSI LOAD MODEL & STOPWORDS (CACHED)

@st.cache_resource
def load_components():
    model_dir = os.path.join(os.path.dirname(__file__), "Output", "models")
    vec_path = os.path.join(model_dir, "skipgram_vectors.pkl")
    nb_path = os.path.join(model_dir, "naive_bayes_model.pkl")
    cls_path = os.path.join(model_dir, "classes.pkl")

    if not os.path.exists(vec_path) or not os.path.exists(nb_path):
        raise FileNotFoundError(f"File model tidak ditemukan di {model_dir}. Pastikan model telah disimpan.")

    sg_bundle = joblib.load(vec_path)
    model_sg_wv = sg_bundle["vectors"]
    vector_size = sg_bundle.get("vector_size", 100)
    model_nb = joblib.load(nb_path)
    classes = joblib.load(cls_path) if os.path.exists(cls_path) else list(model_nb.classes_)

    # Stopwords Sastrawi
    factory = StopWordRemoverFactory()
    stopword_set = set(factory.get_stop_words())

    return model_sg_wv, vector_size, model_nb, classes, stopword_set

try:
    model_sg_wv, vector_size, model_nb, classes, stopword_set = load_components()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_err = str(e)


# EKSTRAKSI ARTIKEL DENGAN TRAFILATURA
def fetch_news_from_url(url: str):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Halaman berita tidak berhasil diambil dari URL.")

    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title.strip() if metadata and metadata.title else ""
    text_content = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True
    )

    if not text_content or len(text_content.strip()) < 30:
        raise ValueError("Trafilatura tidak menemukan isi artikel yang mencukupi.")

    return title, text_content.strip()

def preprocess_text(text: str, stopwords: set):
    # 1. Pembersihan karakter non-alfabet
    clean = re.sub(r"[^a-zA-Z\s]", " ", text)
    # 2. Case folding (lowercase)
    clean = clean.lower().strip()
    # 3. Tokenisasi & Stopword removal Sastrawi
    tokens = [w for w in clean.split() if w not in stopwords and len(w) > 1]
    return tokens

def get_mean_vector(tokens, word_vectors_dict, dim=100):
    word_vecs = [word_vectors_dict[w] for w in tokens if w in word_vectors_dict]
    if len(word_vecs) == 0:
        return np.zeros(dim)
    return np.mean(word_vecs, axis=0)


# TAMPILAN HEADER UTAMA

st.markdown('<div class="main-title"> Klasifikasi Teks Berita </div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Prediksi kategori artikel berita (<b>Sport</b> atau <b>Finance</b>) ',
    unsafe_allow_html=True
)

if not model_loaded:
    st.error(f"Gagal memuat model: {load_err}")
    st.stop()

# Sidebar: Info Model
with st.sidebar:
    st.header("⚙️ Informasi Singkat")
    st.markdown("""
    **Mesin 1: Vektorisasi Teks**
    - Dimensi: **100 Fitur**
    - Model: **Skip-gram Word2Vec**

    **Mesin 2: Klasifikasi Target**
    - Model: **Gaussian Naive Bayes**
    - Akurasi Model: **97.50%**
    - Kategori Target: **Sport** & **Finance**
    """.format(len(model_sg_wv)))
    
    st.divider()
    st.caption("PPW 2026")


# INPUT FORM: LINK BERITA
st.write("Copy dan paste link berita (misalnya Detik Sport / Detik Finance atau berita yang lain):")
url_input = st.text_input(
    "URL Berita:",
    placeholder="https://sport.detik.com/... atau https://finance.detik.com/...",
    key="news_url"
)
btn_predict_url = st.button("Ambil Berita & Prediksi Berita", type="primary", key="btn_url")

news_title = ""
news_text = ""

# Trigger Prediksi
if btn_predict_url and url_input.strip():
    with st.spinner("Mengambil konten berita dari link via HTTP..."):
        try:
            news_title, news_text = fetch_news_from_url(url_input.strip())
            if not news_text or len(news_text) < 30:
                st.warning("Gagal menemukan teks artikel yang mencukupi dari link tersebut. Silakan coba link berita lain.")
        except Exception as err:
            st.error(f"Gagal mengambil artikel dari link: {err}")


# PROSES PREDIKSI & VISUALISASI PERSENTASE

if news_text:
    tokens = preprocess_text(news_text, stopword_set)

    if len(tokens) == 0:
        st.warning("Teks tidak menghasilkan token kata yang valid setelah pembersihan.")
    else:
        in_vocab_tokens = [t for t in tokens if t in model_sg_wv]
        if len(in_vocab_tokens) < 3:
            st.warning(
                "Kata-kata berita tidak cukup cocok dengan kosakata model yang dilatih. "
                "Ini sering menyebabkan prediksi mengarah ke kelas dominan (misalnya Finance) karena vektor dokumen menjadi terlalu lemah. "
                f"Token cocok model: {len(in_vocab_tokens)}/{len(tokens)}."
            )
            if news_title:
                title_tokens = preprocess_text(news_title, stopword_set)
                title_in_vocab = [t for t in title_tokens if t in model_sg_wv]
                if len(title_in_vocab) >= 3:
                    st.info("Fallback ke judul artikel karena teks isi terlalu sedikit cocok dengan kosakata model.")
                    tokens = title_tokens
                    in_vocab_tokens = title_in_vocab

        if len(in_vocab_tokens) == 0:
            st.error("Tidak ada token yang cocok dengan kosakata model. Prediksi tidak dapat dilakukan secara valid.")
        else:
            # Mesin 1: Ekstraksi Vektor Dokumen Skip-gram
            doc_vec = get_mean_vector(tokens, model_sg_wv, dim=vector_size)

            # Mesin 2: Prediksi Naive Bayes
            doc_vec_2d = doc_vec.reshape(1, -1)
            pred_label = model_nb.predict(doc_vec_2d)[0]
            pred_proba = model_nb.predict_proba(doc_vec_2d)[0]

            # Hitung Persentase Probabilitas
            prob_dict = {cls: float(prob) * 100 for cls, prob in zip(classes, pred_proba)}
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

            st.divider()

            # Tampilan Hasil Prediksi Target
            col_res1, col_res2 = st.columns([1.2, 1])

            with col_res1:
                st.subheader("Hasil Prediksi Target")

                badge_class = "status-sport" if pred_label.lower() == "sport" else "status-finance"
                icon = "🏸" if pred_label.lower() == "sport" else "📈"

                st.markdown(
                    f'<div class="{badge_class}">{icon} {pred_label.upper()}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(f"**Tingkat Keyakinan:** `{prob_dict[pred_label]:.2f}%`")

                if news_title:
                    st.markdown(f"**Judul Artikel:** *{news_title}*")

                st.caption(f"Jumlah token kata yang dihitung: {len(tokens)} token | Vektor: {doc_vec.shape[0]} dimensi")

            with col_res2:
                st.subheader("📊 Persentase Probabilitas Target")
                for cls_name, pct in sorted_probs:
                    pct = float(pct)
                    st.write(f"**{cls_name}**: `{pct:.2f}%`")
                    st.progress(min(pct / 100.0, 1.0))
