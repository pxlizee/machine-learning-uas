import streamlit as st
import joblib
import re
import os
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. CONFIG & LOAD MODEL
# ==========================================
st.set_page_config(
    page_title="Gojek Sentiment Dashboard & Analyzer",
    page_icon="🍏",
    layout="wide" # Mengubah ke mode wide agar visualisasi terlihat luas
)

@st.cache_resource
def load_assets():
    vectorizer = joblib.load("data/processed/tfidf.pkl")
    model = joblib.load("data/processed/sentiment_model.pkl")
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())
    return vectorizer, model, stopwords

try:
    vectorizer, model, stopwords = load_assets()
except FileNotFoundError:
    st.error("❌ File model tidak ditemukan! Jalankan skrip evaluasi terlebih dahulu.")
    st.stop()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stopwords]
    return " ".join(words)

# ==========================================
# 2. HEADER APLIKASI
# ==========================================
st.title("📱 Gojek Sentiment Analytics Platform")
st.caption("Aplikasi Deployment Model Naïve Bayes & Visualisasi Dataset Ulasan Gojek")
st.markdown("---")

# Membuat Navigasi Tab
tab1, tab2 = st.tabs(["🔮 Prediksi Ulasan Baru", "📊 Dashboard & Visualisasi"])

# ==========================================
# TAB 1: PREDIKSI ULASAN
# ==========================================
with tab1:
    st.header("Uji Coba Prediksi Sentimen")
    st.write("Masukkan teks ulasan baru di bawah ini untuk melihat bagaimana model Naïve Bayes mengklasifikasikannya.")
    
    user_input = st.text_area(
        "Masukkan teks ulasan Gojek:",
        placeholder="Ketik di sini...",
        height=120
    )

    if st.button("Analisis Sentimen", type="primary"):
        if user_input.strip() == "":
            st.warning("⚠️ Silakan ketik ulasan terlebih dahulu!")
        else:
            cleaned_text = preprocess_text(user_input)
            text_vector = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_vector)[0]
            
            st.markdown("### Hasil Analisis:")
            if prediction == "positif":
                st.success(f"🟢 **Sentimen: POSITIF**\n\nUlasan terdeteksi bernilai kepuasan pelanggan.")
            elif prediction == "netral":
                st.info(f"🟡 **Sentimen: NETRAL**\n\nUlasan terdeteksi bernilai biasa saja / berimbang.")
            else:
                st.error(f"🔴 **Sentimen: NEGATIF**\n\nUlasan terdeteksi bernilai keluhan atau kekecewaan.")

# ==========================================
# TAB 2: VISUALISASI DATASET
# ==========================================
with tab2:
    st.header("Analisis & Statistik Dataset Global")
    st.write("Berikut adalah hasil pemrosesan dan visualisasi dari seluruh dataset ulasan Gojek yang telah diolah.")
    
    # Menampilkan Grafik Distribusi (Bar & Pie Chart berdampingan)
    st.subheader("1. Distribusi Kategori Sentimen")
    if os.path.exists('data/processed/distribusi_sentimen.png'):
        st.image('data/processed/distribusi_sentimen.png', caption='Proporsi dan Jumlah Data per Sentimen', use_container_width=True)
    else:
        st.info("Grafik distribusi belum di-generate. Jalankan skrip 04_visualisasi.py terlebih dahulu.")
        
    st.markdown("---")
    
    # Menampilkan Word Cloud Positif vs Negatif berdampingan
    st.subheader("2. Awan Kata (Word Cloud) Berdasarkan Sentimen")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='text-align: center; color: #2ecc71;'>Sentimen Positif</h4>", unsafe_allow_html=True)
        if os.path.exists('data/processed/wordcloud_positif.png'):
            st.image('data/processed/wordcloud_positif.png', use_container_width=True)
        else:
            st.caption("Wordcloud positif tidak ditemukan.")
            
    with col2:
        st.markdown("<h4 style='text-align: center; color: #e74c3c;'>Sentimen Negatif</h4>", unsafe_allow_html=True)
        if os.path.exists('data/processed/wordcloud_negatif.png'):
            st.image('data/processed/wordcloud_negatif.png', use_container_width=True)
        else:
            st.caption("Wordcloud negatif tidak ditemukan.")

    st.markdown("---")
    st.subheader("3. Performa Evaluasi Model")
    if os.path.exists('data/processed/confusion_matrix.png'):
        st.image('data/processed/confusion_matrix.png', caption='Confusion Matrix Evaluasi Naïve Bayes', width=600)