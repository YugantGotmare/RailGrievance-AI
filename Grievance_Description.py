import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from prompt import *

def Grievance_Description(description):
    try:
        # Create the FAISS index with your grievance types
        embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vectorstore = FAISS.from_texts(grievance_types, embedding)  

        # Search for similar grievance type
        similarity_results = vectorstore.similarity_search(description, k=3)

        # Use the top 3 results to feed into the classification prompt
        top_similarities = [result.page_content for result in similarity_results]

        # Join the top similarities into the prompt
        combined_prompt = f"Top similar categories: {', '.join(top_similarities)}\n\nGrievance Description: {description}"

        matched_type = safe_get_type_of_Grievance_Description(combined_prompt)
        return matched_type
    except Exception as e:
        st.error(f"Error in Grievance_Description: {e}")
        return "security"