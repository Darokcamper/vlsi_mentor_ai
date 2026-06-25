import os
import time
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

# Underlying Groq model instance
raw_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0.8
)

# Wrapper to handle Groq API rate limits (TPM/RPM 429 errors) with retry backoff
class RateLimitedChatGroq:
    def __init__(self, chat_model: ChatGroq, max_retries: int = 6, initial_backoff: float = 4.0):
        self.chat_model = chat_model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def invoke(self, *args, **kwargs):
        retries = 0
        backoff = self.initial_backoff
        while True:
            try:
                return self.chat_model.invoke(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "limit" in err_str:
                    retries += 1
                    if retries > self.max_retries:
                        raise e
                    # Log rate limit event to stdout and streamlit if active
                    msg = f"Groq Rate limit hit. Retrying in {backoff:.1f}s... (Attempt {retries}/{self.max_retries})"
                    print(msg)
                    try:
                        st.toast(msg, icon="⏳")
                    except Exception:
                        pass
                    time.sleep(backoff)
                    backoff *= 1.5
                else:
                    raise e

    def stream(self, *args, **kwargs):
        retries = 0
        backoff = self.initial_backoff
        while True:
            try:
                return self.chat_model.stream(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if ("429" in err_str or "rate" in err_str or "limit" in err_str) and retries < self.max_retries:
                    retries += 1
                    # Log rate limit event to stdout and streamlit if active
                    msg = f"Groq Rate limit hit (streaming). Retrying in {backoff:.1f}s... (Attempt {retries}/{self.max_retries})"
                    print(msg)
                    try:
                        st.toast(msg, icon="⏳")
                    except Exception:
                        pass
                    time.sleep(backoff)
                    backoff *= 1.5
                else:
                    raise e

# Export wrapped model
llm = RateLimitedChatGroq(raw_llm)