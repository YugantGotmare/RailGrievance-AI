import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
from get_embeddings import *
from get_text_chunks import *
from get_pdf_text import *
from Grievance_Description import *
from prompt import *


load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("Google API Key not found. Please set the GOOGLE_API_KEY in your .env file.")
else:
    genai.configure(api_key=api_key)

def main():
    # Set page configuration
    st.set_page_config(layout="wide")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .title {
            font-size: 2.5em;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 1.2em;
            color: #34495e;
            text-align: center;
            margin-bottom: 40px;
        }
        .card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Title and subtitle
    st.markdown("<h1 style='text-align: center;'>RailGrievance AI 📄</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload a ticket PDF to extract details.</p>", unsafe_allow_html=True)

    # Initialize session state for details and grievance category
    if 'details_dict' not in st.session_state:
        st.session_state.details_dict = {
            "PNR Number": "N/A",
            "Incident Date": str(datetime.now().date()),
            "Train No./Name": "N/A",
            "Seat Number": "N/A",
        }
    if 'grievance_category' not in st.session_state:
        st.session_state.grievance_category = ""

    # File uploader
    document = st.file_uploader("Upload a PDF", type=['pdf'], help="Upload a PDF ticket file (max size: 5MB). Ensure it contains relevant details like PNR Number, Train Name, etc.")

    if st.button('Submit and Process'):
        if document is not None:
            with st.spinner('Processing...'):
                text = get_pdf_text(document)
                if text:
                    details = extract_ticket_details(text)
                    
                    for line in details.split('\n'):
                        if 'PNR Number' in line:
                            st.session_state.details_dict['PNR Number'] = line.split(':')[1].strip()
                        elif 'Train No./Name' in line:
                            st.session_state.details_dict['Train No./Name'] = line.split(':')[1].strip()
                        elif 'Seat Number' in line:
                            st.session_state.details_dict['Seat Number'] = line.split(':')[1].strip()
                else:
                    st.error("No text extracted from the PDF. Please check the file.")
        else:
            st.error('Please upload a PDF file.')

    # Extracted details section
    with st.expander("Extracted Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            pnr_number = st.text_input("PNR Number", value=st.session_state.details_dict['PNR Number'], placeholder="e.g., 1234567890", key="pnr_number_input")
        with col2:
            incident_date = st.date_input("Incident Date", value=datetime.now(), key="incident_date_input")
        with col3:
            train_no_name = st.text_input("Train No./Name", value=st.session_state.details_dict['Train No./Name'], placeholder="e.g., Express Train", key="train_no_name_input")

        seat_number = st.text_input("Seat Number", value=st.session_state.details_dict['Seat Number'], placeholder="e.g., 12A", key="seat_number_input")

        grievance_description = st.text_area("Enter Grievance Description", placeholder="Describe the grievance here", key="grievance_description_input")

        if st.button("Match Grievance Type"):
            if grievance_description:
                try:
                    st.session_state.grievance_category = Grievance_Description(grievance_description)
                    if st.session_state.grievance_category == "security":
                        st.warning("This grievance may involve sensitive or safety-related issues. It has been categorized as 'security'. Please handle with appropriate care and follow relevant protocols.")
                    else:
                        st.success(f"Matched Grievance Category: {st.session_state.grievance_category}")
                except Exception as e:
                    st.error(f"Error matching grievance type: {e}")
                    st.session_state.grievance_category = "security"
                    st.warning("Due to an error, this grievance has been categorized as 'security' by default. Please review manually.")
            else:
                st.warning("Please enter a grievance description.")

        if st.session_state.grievance_category:
            st.success(f"Type Grievance: {st.session_state.grievance_category}")

        if st.button("Submit Details"):
            st.success(f"Details Submitted:\n"
                       f"PNR: {pnr_number}\n"
                       f"Incident Date: {incident_date}\n"
                       f"Train No./Name: {train_no_name}\n"
                       f"Seat Number: {seat_number}\n"
                       f"Grievance Description: {grievance_description}\n"
                       f"Category: {st.session_state.grievance_category}")

if __name__ == '__main__':
    main()