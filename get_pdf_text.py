from PyPDF2 import PdfReader
import streamlit as st

def get_pdf_text(pdf):
    text = ""
    try:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text