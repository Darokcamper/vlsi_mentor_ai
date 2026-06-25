import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set the GROQ_API_KEY environment variable or Streamlit secret.")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0.8
)