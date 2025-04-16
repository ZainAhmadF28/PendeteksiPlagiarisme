import streamlit as st
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Pastikan Anda telah mengunduh stopwords dari nltk
nltk.download('stopwords')

# Fungsi untuk mengekstrak teks dari PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Fungsi untuk membaca teks dari file txt
def extract_text_from_txt(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    return text

# Fungsi untuk menghitung cosine similarity antara dua teks
def calculate_cosine_similarity(text1, text2):
    # Stopwords untuk bahasa Indonesia
    stopwords_indonesia = stopwords.words('indonesian')
    
    # Menggunakan TfidfVectorizer untuk memroses teks
    tfidf = TfidfVectorizer(stop_words=stopwords_indonesia)
    tfidf_matrix = tfidf.fit_transform([text1, text2])
    
    # Menghitung cosine similarity antara kedua teks
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity_matrix[0][0]

# Header
st.title("Aplikasi Deteksi Plagiarisme")
st.subheader("Deteksi Plagiarisme dengan mudah menggunakan file teks atau PDF")

# Sidebar untuk pilihan
st.sidebar.header("Pengaturan")
file_type = st.sidebar.selectbox(
    "Pilih Tipe File untuk Diuji:",
    ("Teks", "PDF")
)

# Input Referensi Sumber
reference_file_type = st.sidebar.selectbox(
    "Pilih Tipe File Referensi Sumber:",
    ("Teks", "PDF")
)

# Input Teks atau Upload File untuk Referensi Sumber
if reference_file_type == "Teks":
    st.write("Masukkan teks referensi sumber:")
    reference_text = st.text_area("Masukkan Teks Referensi")
elif reference_file_type == "PDF":
    st.write("Unggah file PDF sebagai referensi sumber:")
    reference_uploaded_file = st.file_uploader("Pilih PDF Referensi", type="pdf")

# Input Teks atau Upload File untuk Diuji
if file_type == "Teks":
    st.write("Masukkan teks yang akan diuji plagiarisme:")
    text_input = st.text_area("Masukkan Teks")
elif file_type == "PDF":
    st.write("Unggah file PDF yang ingin diuji plagiarisme:")
    uploaded_file = st.file_uploader("Pilih PDF untuk Diuji", type="pdf")

# Tombol untuk Memulai Deteksi
if st.button('Mulai Deteksi Plagiarisme'):
    if reference_file_type == "Teks" and reference_text and file_type == "Teks" and text_input:
        st.write("Memulai deteksi plagiarisme pada teks...")
        # Hitung similarity
        similarity_score = calculate_cosine_similarity(text_input, reference_text)
        
        # Menampilkan hasil
        if similarity_score > 0.8:
            st.warning(f"Teks sangat mirip dengan referensi! Similarity score: {similarity_score:.2f}")
        else:
            st.success(f"Teks tidak terlalu mirip dengan referensi. Similarity score: {similarity_score:.2f}")
    
    elif reference_file_type == "PDF" and reference_uploaded_file and file_type == "Teks" and text_input:
        st.write("Memulai deteksi plagiarisme pada file PDF referensi dan teks...")
        
        # Ekstrak teks dari file PDF referensi
        reference_pdf_text = extract_text_from_pdf(reference_uploaded_file)
        
        # Hitung similarity
        similarity_score = calculate_cosine_similarity(text_input, reference_pdf_text)
        
        # Menampilkan hasil
        if similarity_score > 0.8:
            st.warning(f"Teks sangat mirip dengan referensi! Similarity score: {similarity_score:.2f}")
        else:
            st.success(f"Teks tidak terlalu mirip dengan referensi. Similarity score: {similarity_score:.2f}")
    
    elif reference_file_type == "Teks" and reference_text and file_type == "PDF" and uploaded_file:
        st.write("Memulai deteksi plagiarisme pada file PDF referensi dan file PDF yang diunggah...")
        
        # Ekstrak teks dari file PDF referensi dan file PDF yang diunggah
        reference_pdf_text = extract_text_from_pdf(reference_uploaded_file)
        uploaded_pdf_text = extract_text_from_pdf(uploaded_file)
        
        # Hitung similarity
        similarity_score = calculate_cosine_similarity(uploaded_pdf_text, reference_pdf_text)
        
        # Menampilkan hasil
        if similarity_score > 0.8:
            st.warning(f"Teks dalam PDF sangat mirip dengan referensi! Similarity score: {similarity_score:.2f}")
        else:
            st.success(f"Teks dalam PDF tidak terlalu mirip dengan referensi. Similarity score: {similarity_score:.2f}")
    
    elif reference_file_type == "PDF" and reference_uploaded_file and file_type == "PDF" and uploaded_file:
        st.write("Memulai deteksi plagiarisme pada kedua file PDF...")
        
        # Ekstrak teks dari kedua file PDF
        reference_pdf_text = extract_text_from_pdf(reference_uploaded_file)
        uploaded_pdf_text = extract_text_from_pdf(uploaded_file)
        
        # Hitung similarity
        similarity_score = calculate_cosine_similarity(uploaded_pdf_text, reference_pdf_text)
        
        # Menampilkan hasil
        if similarity_score > 0.8:
            st.warning(f"File PDF sangat mirip dengan referensi! Similarity score: {similarity_score:.2f}")
        else:
            st.success(f"File PDF tidak terlalu mirip dengan referensi. Similarity score: {similarity_score:.2f}")
    
    else:
        st.warning("Silakan masukkan teks referensi atau unggah file PDF terlebih dahulu.")

# Menampilkan footer atau catatan
st.markdown("### Tentang Aplikasi")
st.write("Aplikasi ini memungkinkan deteksi plagiarisme menggunakan teks atau PDF dengan berbagai metode perbandingan teks.")
