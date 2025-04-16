import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

import spacy.cli
spacy.cli.download("xx_ent_wiki_sm")  # tambahkan ini

# Muat model bahasa spaCy
nlp = spacy.load("xx_ent_wiki_sm")

# Menambahkan sentencizer ke pipeline spaCy dengan nama string
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Fungsi ekstraksi PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Fungsi ekstraksi dari txt
def extract_text_from_txt(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    return text

# Fungsi cosine similarity
def calculate_cosine_similarity(text1, text2):
    stopwords_indonesia = stopwords.words('indonesian')
    tfidf = TfidfVectorizer(stop_words=stopwords_indonesia)
    tfidf_matrix = tfidf.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]

# Tokenisasi kalimat menggunakan spaCy
def sent_tokenize_spacy(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# Highlight bagian mirip
def highlight_similar_sentences(text_input, reference_text, threshold=0.8):
    sentences_input = sent_tokenize_spacy(text_input)  # Menggunakan spaCy untuk tokenisasi kalimat
    sentences_ref = sent_tokenize_spacy(reference_text)  # Menggunakan spaCy untuk tokenisasi kalimat

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

# UI Streamlit
st.title("Aplikasi Deteksi Plagiarisme")
st.subheader("Deteksi Plagiarisme dan Lihat Bagian yang Mirip 🔍")

st.sidebar.header("Pengaturan")
file_type = st.sidebar.selectbox("Pilih Tipe File untuk Diuji:", ("Teks", "PDF"))
reference_file_type = st.sidebar.selectbox("Pilih Tipe File Referensi:", ("Teks", "PDF"))

# Input referensi
reference_text = ""
reference_uploaded_file = None
if reference_file_type == "Teks":
    reference_text = st.text_area("Masukkan Teks Referensi:")
elif reference_file_type == "PDF":
    reference_uploaded_file = st.file_uploader("Unggah PDF Referensi", type="pdf")

# Input dokumen uji
text_input = ""
uploaded_file = None
if file_type == "Teks":
    text_input = st.text_area("Masukkan Teks yang Akan Diuji:")
elif file_type == "PDF":
    uploaded_file = st.file_uploader("Unggah PDF untuk Diuji", type="pdf")

# Tombol deteksi
if st.button('🔍 Mulai Deteksi Plagiarisme'):
    if reference_file_type == "PDF" and reference_uploaded_file:
        reference_text = extract_text_from_pdf(reference_uploaded_file)
    if file_type == "PDF" and uploaded_file:
        text_input = extract_text_from_pdf(uploaded_file)

    if text_input and reference_text:
        similarity_score = calculate_cosine_similarity(text_input, reference_text)
        st.subheader(f"📊 Skor Kemiripan: {similarity_score:.2f}")
        if similarity_score > 0.8:
            st.warning("⚠️ Teks sangat mirip dengan referensi!")
        else:
            st.success("✅ Teks tidak terlalu mirip dengan referensi.")

        # Highlight bagian yang mirip
        st.markdown("### ✨ Bagian yang Mirip:")
        similar_sentences = highlight_similar_sentences(text_input, reference_text)
        if similar_sentences:
            for s in similar_sentences:
                st.markdown(s)
        else:
            st.info("Tidak ada bagian yang sangat mirip ditemukan berdasarkan threshold.")

    else:
        st.warning("Mohon masukkan teks dan referensi yang valid.")

# Footer
st.markdown("### ℹ️ Tentang Aplikasi")
st.write("Aplikasi ini menggunakan cosine similarity dan TF-IDF untuk mendeteksi kemiripan dokumen dan menunjukkan bagian yang mirip antar teks.")
