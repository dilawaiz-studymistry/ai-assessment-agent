import streamlit as st
from google import genai
from google.genai import types
import fitz  # PyMuPDF
import json
import re

# Page Setup
st.set_page_config(page_title="Academic Marking Agent", layout="wide")
st.title("🎓 AI Academic Marking Agent")
st.markdown("Upload a **Student Answer Script** and a **Marking Scheme** to get visual score annotations and an Excel-ready summary matrix.")

# Fetch API Key from Secrets
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

# System Prompt
SYSTEM_INSTRUCTIONS = """
You are a precise Academic Data Formatting Agent.
Analyze a Student Answer Script against a Marking Scheme.

OUTPUT REQUIREMENTS:
1. Provide a clean Markdown report with score feedback and an Excel matrix table.
2. AT THE VERY END OF YOUR RESPONSE, provide a raw JSON block wrapped strictly in ```json ... ``` containing page-by-page overlay marks.

JSON Format required:
```json
[
  {
    "page": 1,
    "score_text": "Q1: 4/5 (Missing arrows)",
    "x": 50,
    "y": 100
  },
  {
    "page": 1,
    "score_text": "Q2: 5/5",
    "x": 50,
    "y": 300
  }
]
