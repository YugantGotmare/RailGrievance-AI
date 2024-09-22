from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from get_text_chunks import *
from get_embeddings import *

grievance_types = [
    "medical assistance", 
    "security", 
    "divyangjan facilities",
    "facilities for women with special needs", 
    "electrical equipment",
    "coach - cleanliness", 
    "punctuality", 
    "water availability",
    "catering & vending service", 
    "staff behaviour", 
    "corruption/bribery",
    "bed roll", 
    "miscellaneous"
]

def get_conversational_chain():
    prompt_temp = '''
    Extract the following details from the ticket:
    - PNR Number
    - Train No./Name
    - Seat Number (seat number can be written as Booking Status or Current Status)
    - Incident Date
    Context:\n{context}\n
    '''
    prompt = PromptTemplate(
        template=prompt_temp,
        input_variables=['context']
    )

    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.5)
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def safe_get_type_of_Grievance_Description(description):
    prompt_temp = f'''
    These are the types of grievances:
    {', '.join(grievance_types)}
    
    Match the following grievance description with one of these types.
    If the description involves sensitive topics like harassment or safety concerns,
    categorize it as "security" or the most appropriate category without repeating sensitive details.
    
    Grievance Description: {description}
    
    Respond with only the matched grievance type. If no match is found, respond with "miscellaneous".
    '''
    
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    try:
        response = model.invoke(prompt_temp)
        return response.content.strip()
    except google.api_core.exceptions.GoogleAPIError as e:
        if "safety" in str(e).lower():
            return "security"  # Default to "security" for safety-related issues
        else:
            raise e


def extract_ticket_details(text):
    chunks = get_text_chunks(text)
    get_embeddings(chunks)
    
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search('Extract the following details from the ticket: PNR Number, Train No./Name, Seat Number, Incident Date')
    chain = get_conversational_chain()
    response = chain(
        {'input_documents': docs, 'question': 'Extract PNR Number, Train No./Name, Seat Number, and Incident Date from ticket'},
        return_only_outputs=True
    )
    return response['output_text']