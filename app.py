import streamlit as st
import os
import tempfile
from index import index_documents
from model import search
from qa import answer_question

# Custom CSS styles
st.markdown("""
    <style>
        html, body {
            background-color: #f9f9ff;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            font-size: 3em;
            color: #4a00e0;
        }
        .stTextInput > div > div > input {
            background-color: #ffffff;
            border: 2px solid #6c63ff;
            border-radius: 10px;
            padding: 0.75em;
            font-size: 1em;
        }
        .stFileUploader {
            background-color: #f1f1fa;
            border: 2px dashed #6c63ff;
            padding: 1.5em;
            border-radius: 10px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6c63ff, #4a00e0);
            color: white;
            font-weight: bold;
            padding: 0.6em 1.5em;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #4a00e0, #6c63ff);
            color: #fff;
        }
        .result-box {
            background-color: #ffffff;
            padding: 1.25em;
            border-left: 5px solid #6c63ff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
            margin-bottom: 1em;
        }
        .answer-box {
            background-color: #e8e8ff;
            padding: 1.5em;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.25em;
            color: #333;
            box-shadow: 0 0 10px rgba(108, 99, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

# Set page config
st.set_page_config(page_title="📚 DocuSavvy", layout="wide")

# Header
st.markdown("""
    <h1 style='text-align: center;'>📚 DocuSavvy</h1>
    <p style='text-align: center; font-size: 1.1em;'>Your Offline AI Document Assistant — Secure. Smart. Sleek.</p>
""", unsafe_allow_html=True)

# File uploader
uploaded_files = st.file_uploader(
    "📤 Upload your documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Main App Logic
if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in uploaded_files:
            with open(os.path.join(tmpdir, f.name), "wb") as out:
                out.write(f.read())

        st.success("✅ Documents uploaded and processed!")

        with st.spinner("🔍 Indexing your documents..."):
            index, chunks, filenames = index_documents(tmpdir)
        st.success("📚 Documents indexed and ready!")

        st.markdown("### 🔎 Ask something about your documents:")
        query = st.text_input("", placeholder="e.g. What is the father name of Gaurav?", label_visibility="collapsed")

        if st.button("💬 Ask"):
            if query.strip():
                with st.spinner("🤖 Thinking..."):
                    results = search(query, index, chunks, filenames)
                    context = " ".join([chunk for chunk, _ in results])
                    answer = answer_question(query, context)

                st.markdown("### 🧠 Answer")
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

                st.markdown("### 🧾 Top Matches")
                for text, fname in results:
                    st.markdown(f"""
                        <div class='result-box'>
                            <b>📄 File:</b> <code>{fname}</code><br><br>
                            <i>{text}</i>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a question before clicking 'Ask'.")
else:
    st.info("👆 Upload some documents to get started.")