import streamlit as st
import os
import re
import joblib
import numpy as np
import urllib.request
import html
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

# -------------------------------------------------------------
# FUNGSI LOAD MODEL & STOPWORDS (CACHED)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# FUNGSI SCRAPING MURNI (TANPA BEAUTIFULSOUP)
# Menggunakan urllib.request bawaan Python & Regular Expressions
# -------------------------------------------------------------
def fetch_news_from_url(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=12) as response:
        raw_html = response.read().decode("utf-8", errors="ignore")

    # 1. Ekstraksi Judul
    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", raw_html, re.IGNORECASE | re.DOTALL)
    if not title_match:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
    
    title = ""
    if title_match:
        title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
        title = html.unescape(title)

    # 2. Bersihkan tag <script> dan <style>
    clean_html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)

    # 3. Ekstraksi isi semua tag paragraf <p>...</p>
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", clean_html, flags=re.IGNORECASE | re.DOTALL)
    extracted_texts = []
    
    for p in paragraphs:
        text_p = re.sub(r"<[^>]+>", " ", p) # Buang tag anak di dalam <p>
        text_p = html.unescape(text_p).strip()
        text_p = re.sub(r"\s+", " ", text_p)
        # Saring hanya teks paragraf bermakna (hindari disclaimer pendek / copyright)
        if len(text_p) > 25:
            extracted_texts.append(text_p)

    text_content = " ".join(extracted_texts)
    
    # Fallback jika tidak menemukan tag <p> yang cukup
    if len(text_content) < 50:
        text_content = re.sub(r"<[^>]+>", " ", clean_html)
        text_content = html.unescape(text_content).strip()
        text_content = re.sub(r"\s+", " ", text_content)

    return title, text_content

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

# -------------------------------------------------------------
# TAMPILAN HEADER UTAMA
# -------------------------------------------------------------
st.markdown('<div class="main-title">📰 Klasifikasi Teks Berita Daring</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Prediksi kategori artikel berita (<b>Sport</b> atau <b>Finance</b>) '
    'menggunakan 2 mesin: <span class="engine-badge badge-sg">Mesin 1: Skip-gram Embedding</span> '
    '<span class="engine-badge badge-nb">Mesin 2: Gaussian Naive Bayes</span></div>',
    unsafe_allow_html=True
)

if not model_loaded:
    st.error(f"Gagal memuat model: {load_err}")
    st.stop()

# Sidebar: Info Model
with st.sidebar:
    st.header("⚙️ Informasi 2 Mesin")
    st.markdown("""
    **Mesin 1: Vektorisasi Teks**
    - Arsitektur: **Word2Vec Skip-gram (`sg=1`)**
    - Dimensi: **100 Fitur**
    - Pooling: **Mean Word Vector**
    - Kosakata Model: `{}` kata unik
    
    **Mesin 2: Klasifikasi Target**
    - Model: **Gaussian Naive Bayes**
    - Akurasi Model: **97.50%**
    - Kategori Target: **Sport** & **Finance**
    """.format(len(model_sg_wv)))
    
    st.divider()
    st.caption("Praktek Pembelajaran Web Mining - PPW 2026")

# -------------------------------------------------------------
# INPUT FORM: LINK BERITA / TEKS MANUAL
# -------------------------------------------------------------
input_tab1, input_tab2 = st.tabs(["🔗 Prediksi dari Link Berita (URL)", "✍️ Masukkan Teks Manual"])

with input_tab1:
    st.write("Copy dan paste link berita (misalnya Detik Sport / Detik Finance):")
    url_input = st.text_input(
        "URL Berita:",
        placeholder="https://sport.detik.com/... atau https://finance.detik.com/...",
        key="news_url"
    )
    btn_predict_url = st.button("🚀 Ambil Berita & Prediksi Target", type="primary", key="btn_url")

with input_tab2:
    manual_text = st.text_area(
        "Ketik / Tempel Isi Berita Langsung:",
        placeholder="Tempel teks artikel berita di sini...",
        height=180,
        key="manual_news_text"
    )
    btn_predict_manual = st.button("🚀 Prediksi Teks", type="primary", key="btn_manual")

news_title = ""
news_text = ""

# Trigger Prediksi
if btn_predict_url and url_input.strip():
    with st.spinner("Mengambil konten berita dari link via HTTP..."):
        try:
            news_title, news_text = fetch_news_from_url(url_input.strip())
            if not news_text or len(news_text) < 30:
                st.warning("Gagal menemukan teks artikel yang mencukupi dari link tersebut. Silakan coba link lain atau gunakan tab input manual.")
        except Exception as err:
            st.error(f"Gagal mengambil artikel dari link: {err}")

elif btn_predict_manual and manual_text.strip():
    news_title = "Input Teks Manual"
    news_text = manual_text.strip()

# -------------------------------------------------------------
# PROSES PREDIKSI & VISUALISASI PERSENTASE
# -------------------------------------------------------------
if news_text:
    tokens = preprocess_text(news_text, stopword_set)
    
    if len(tokens) == 0:
        st.warning("Teks tidak menghasilkan token kata yang valid setelah pembersihan.")
    else:
        # Mesin 1: Ekstraksi Vektor Dokumen Skip-gram
        doc_vec = get_mean_vector(tokens, model_sg_wv, dim=vector_size)
        
        # Mesin 2: Prediksi Naive Bayes
        doc_vec_2d = doc_vec.reshape(1, -1)
        pred_label = model_nb.predict(doc_vec_2d)[0]
        pred_proba = model_nb.predict_proba(doc_vec_2d)[0]
        
        # Hitung Persentase Probabilitas
        prob_dict = {cls: prob * 100 for cls, prob in zip(classes, pred_proba)}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        st.divider()
        
        # Tampilan Hasil Prediksi Target
        col_res1, col_res2 = st.columns([1.2, 1])
        
        with col_res1:
            st.subheader("🎯 Hasil Prediksi Target")
            
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

        # Tampilkan detail teks hasil parsing
        with st.expander("🔍 Lihat Teks Berita yang Diproses & Token Kata"):
            st.markdown("**Contoh Teks yang Diekstraksi:**")
            st.write(news_text[:500] + ("..." if len(news_text) > 500 else ""))
            st.markdown("**15 Token Pertama (Setelah Stopword Removal):**")
            st.code(", ".join(tokens[:15]))
