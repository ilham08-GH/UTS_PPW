# 📰 UTS PPW - Klasifikasi Teks Berita (Sport & Finance)

Aplikasi Web Streamlit untuk klasifikasi berita daring berbasis **2 Mesin Pembelajaran Mesin**:
- **Mesin 1 (Vektorisasi/Word Embedding)**: Word2Vec Skip-gram (`sg=1`) dengan Mean Word Vector Pooling (100 dimensi).
- **Mesin 2 (Model Klasifikasi)**: Gaussian Naive Bayes (`GaussianNB`).

---

## 🚀 Fitur Aplikasi
1. **Prediksi dari URL Berita (Link)**: Cukup *copy & paste* tautan artikel berita dari portal berita (Detik.com, dsb.). Artikel diambil langsung via HTTP request dan diekstrak kontennya tanpa ketergantungan library pihak ketiga.
2. **Prediksi Teks Manual**: Masukkan paragraf/teks berita secara langsung.
3. **Target & Persentase Keyakinan**: Menampilkan hasil prediksi kategori (`Sport` / `Finance`) beserta probabilitas persentase untuk masing-masing kelas target.

---

## 🛠️ Cara Menjalankan Lokal

1. **Install Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Jalankan Aplikasi:**
   ```bash
   streamlit run app.py
   ```
