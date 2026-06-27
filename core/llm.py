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
    model="llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.7
)

# List of fallback models on Groq to handle token-per-day (TPD) exhaustion
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

# Wrapper to handle Groq API rate limits (TPM/RPM/TPD 429 errors) with retry backoff and model fallback
class RateLimitedChatGroq:
    def __init__(self, chat_model: ChatGroq, max_retries: int = 6, initial_backoff: float = 4.0):
        self.chat_model = chat_model
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def invoke(self, *args, **kwargs):
        retries = 0
        backoff = self.initial_backoff
        model_idx = 0
        
        while True:
            current_model = MODELS[model_idx]
            self.chat_model.model_name = current_model
            try:
                return self.chat_model.invoke(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                
                # Check for rate limit or quota exceeded
                if "429" in err_str or "rate" in err_str or "limit" in err_str:
                    is_tpd_limit = any(x in err_str for x in ["tpd", "tokens per day", "daily", "daily_limit"])
                    
                    # If TPD limit is hit, rotate immediately to fallback model
                    if is_tpd_limit and model_idx < len(MODELS) - 1:
                        model_idx += 1
                        print(f"\n[Model Fallback] Groq TPD limit hit for {current_model}. Falling back to {MODELS[model_idx]}...")
                        time.sleep(1.0)
                        retries = 0
                        continue
                    
                    retries += 1
                    if retries > self.max_retries:
                        # Try rotating to fallback model as last resort before failing
                        if model_idx < len(MODELS) - 1:
                            model_idx += 1
                            retries = 0
                            print(f"\n[Model Fallback] Max retries hit for {current_model}. Rotating to {MODELS[model_idx]}...")
                            time.sleep(2.0)
                            continue
                        raise e
                    
                    msg = f"Groq Rate limit hit on model {current_model}. Retrying in {backoff:.1f}s... (Attempt {retries}/{self.max_retries})"
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
        model_idx = 0
        
        while True:
            current_model = MODELS[model_idx]
            self.chat_model.model_name = current_model
            try:
                return self.chat_model.stream(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                is_tpd_limit = any(x in err_str for x in ["tpd", "tokens per day", "daily", "daily_limit"])
                
                if is_tpd_limit and model_idx < len(MODELS) - 1:
                    model_idx += 1
                    print(f"\n[Model Fallback] Groq TPD limit hit during stream. Falling back to {MODELS[model_idx]}...")
                    time.sleep(1.0)
                    retries = 0
                    continue
                elif "429" in err_str or "rate" in err_str or "limit" in err_str:
                    retries += 1
                    if retries > self.max_retries:
                        if model_idx < len(MODELS) - 1:
                            model_idx += 1
                            retries = 0
                            print(f"\n[Model Fallback] Max retries hit during stream. Rotating to {MODELS[model_idx]}...")
                            time.sleep(2.0)
                            continue
                        raise e
                    
                    msg = f"Groq Rate limit hit (streaming) on model {current_model}. Retrying in {backoff:.1f}s... (Attempt {retries}/{self.max_retries})"
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