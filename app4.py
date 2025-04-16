import streamlit as st
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download resource hanya jika belum tersedia
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi ekstraksi PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Fungsi cosine similarity
def calculate_cosine_similarity(text1, text2):
    stopwords_indonesia = stopwords.words('indonesian')
    tfidf = TfidfVectorizer(stop_words=stopwords_indonesia)
    tfidf_matrix = tfidf.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]

# Fungsi highlight kalimat mirip
def highlight_similar_sentences(text_input, reference_text, threshold=0.8):
    sentences_input = sent_tokenize(text_input)
    sentences_ref = sent_tokenize(reference_text)

    highlighted_sentences = []
    tfidf = TfidfVectorizer(stop_words=stopwords.words('indonesian'))

    for sentence in sentences_input:
        sims = []
        for ref in sentences_ref:
            tfidf_matrix = tfidf.fit_transform([sentence, ref])
            sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            sims.append(sim_score)
        max_sim = max(sims)
        if max_sim >= threshold:
            highlighted_sentences.append(f"🔸 **{sentence}** (Similarity: {max_sim:.2f})")
    return highlighted_sentences

# Streamlit UI
st.title("Aplikasi Deteksi Plagiarisme")
st.sidebar.header("Pengaturan")

file_type = st.sidebar.selectbox("Pilih Tipe File untuk Diuji:", ("Teks", "PDF"))
reference_file_type = st.sidebar.selectbox("Pilih Tipe File Referensi:", ("Teks", "PDF"))

reference_text = ""
if reference_file_type == "Teks":
    reference_text = st.text_area("Masukkan Teks Referensi:")
elif reference_file_type == "PDF":
    reference_uploaded_file = st.file_uploader("Unggah PDF Referensi", type="pdf")
    if reference_uploaded_file:
        reference_text = extract_text_from_pdf(reference_uploaded_file)

text_input = ""
if file_type == "Teks":
    text_input = st.text_area("Masukkan Teks yang Akan Diuji:")
elif file_type == "PDF":
    uploaded_file = st.file_uploader("Unggah PDF untuk Diuji", type="pdf")
    if uploaded_file:
        text_input = extract_text_from_pdf(uploaded_file)

if st.button('🔍 Mulai Deteksi Plagiarisme'):
    if text_input and reference_text:
        score = calculate_cosine_similarity(text_input, reference_text)
        st.subheader(f"📊 Skor Kemiripan: {score:.2f}")
        if score >= 0.8:
            st.warning("⚠️ Teks sangat mirip dengan referensi!")
        else:
            st.success("✅ Teks tidak terlalu mirip dengan referensi.")
        st.markdown("### ✨ Bagian yang Mirip:")
        result = highlight_similar_sentences(text_input, reference_text)
        if result:
            for item in result:
                st.markdown(item)
        else:
            st.info("Tidak ada bagian yang sangat mirip ditemukan.")
    else:
        st.warning("Mohon isi teks referensi dan teks yang diuji.")
