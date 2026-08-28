import streamlit as st
from google import genai
from google.genai import types
import fitz  # PyMuPDF engine
import json
import re

# 1. Page Configuration
st.set_page_config(page_title="Academic Marking Agent", layout="wide")
st.title("🎓 AI Academic Marking Agent")
st.markdown("Upload a **Student Answer Script** and a **Marking Scheme** to get visual score annotations, an Excel summary matrix, and a stamped PDF.")

# 2. Retrieve Gemini API Key safely from Secrets
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets. Please add it to your Streamlit Cloud configuration.")
    st.stop()

# 3. Complete System Instructions for Detailed Evaluation & Coordinates
SYSTEM_INSTRUCTIONS = """
You are a precise Academic Data Formatting Agent. 

You will analyze two documents uploaded by the user: a handwritten or typed Student Answer Script, and a master Marking Scheme. Your goal is to map out exact scores per question and output structured data meant for spreadsheet population.

CRITICAL OPERATIONAL RULES:
- Do NOT rewrite or replicate any drawings, images, math equations, or text from the student's script. 
- Track student responses solely by their structural identifier (e.g., Q1, Q2, Q3a).
- Read diagrams (such as reaction mechanisms) dynamically. Evaluate if mandatory features (arrows, intermediates, valencies, labels) match the marking scheme, decide the grade, but communicate this strictly as a numerical deduction and a structural textual comment.

OUTPUT REQUIREMENTS:

SECTION 1: VISUAL ANNOTATION LIST
Produce a short list detailing what marks to physically draw over each question on the student's paper. Use this exact syntax:
- [Question Number]: Write "[Score Awarded] / [Max Score]"

SECTION 2: EXCEL SPREADSHEET MATRIX
Provide a clean Markdown table containing your granular evaluations. Format your table with these exact columns:
| Question # | Max Marks | Marks Awarded | Deductions | Deduction Reason / Feedback Summary |

SECTION 3: JSON COORDINATES FOR PDF OVERLAY
AT THE VERY END OF YOUR RESPONSE, provide a raw JSON array containing page-by-page overlay marks for the PDF writer.

Example JSON structure required at the very end:
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

Note: Coordinates (x, y) represent point locations on a standard PDF page (top-left is 0,0). Place score text neatly in open margins near each respective question.
"""

# 4. Helper Function to Draw Red Annotations onto Student PDF Pages
def burn_annotations_to_pdf(pdf_bytes, json_data):
    """Draws red score annotations directly onto the student's PDF pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    try:
        annotations = json.loads(json_data)
        for item in annotations:
            page_num = item.get("page", 1) - 1  # 0-indexed page index
            if 0 <= page_num < len(doc):
                page = doc[page_num]
                text = item.get("score_text", "")
                x = item.get("x", 50)
                y = item.get("y", 100)
                
                # Draw bold red text directly on the PDF
                page.insert_text(
                    fitz.Point(x, y),
                    text,
                    fontsize=12,
                    color=(0.8, 0, 0),  # Red color (RGB)
                    fontname="helv-bold"
                )
    except Exception as e:
        st.warning(f"Could not parse automatic PDF overlay: {e}")

    return doc.tobytes()

# 5. User Interface (File Uploaders)
col1, col2 = st.columns(2)
with col1:
    student_file = st.file_uploader("Upload Student Script (PDF / Image)", type=["pdf", "png", "jpg", "jpeg"])
with col2:
    scheme_file = st.file_uploader("Upload Marking Scheme (PDF / Image)", type=["pdf", "png", "jpg", "jpeg"])

# 6. Marking Execution Logic
if st.button("🚀 Run Assessment Marking"):
    if not student_file or not scheme_file:
        st.warning("Please upload both the Student Answer Script and the Marking Scheme.")
    else:
        with st.spinner("Analyzing handwritten content and building score reports..."):
            try:
                # Initialize GenAI Client
                client = genai.Client(api_key=api_key)
                
                student_bytes = student_file.read()
                scheme_bytes = scheme_file.read()
                
                parts = [
                    types.Part.from_bytes(data=student_bytes, mime_type=student_file.type),
                    types.Part.from_bytes(data=scheme_bytes, mime_type=scheme_file.type),
                    "Grade the student script against the marking scheme according to system instructions."
                ]
                
                # List of reliable models to fall back on if one is busy
                models_to_try = ["gemini-3.5-flash-lite"]
                response = None
                
                for model_name in models_to_try:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=parts,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_INSTRUCTIONS
                            )
                        )
                        if response:
                            break
                    except Exception as e:
                        if "503" in str(e) or "UNAVAILABLE" in str(e):
                            st.warning(f"Model {model_name} is currently busy. Routing request to backup model...")
                            continue
                        else:
                            raise e

                if response:
                    st.success("Marking Complete!")
                    st.markdown("---")
                    
                    # Display full response (Annotation List, Excel Matrix Table, and Feedback)
                    st.markdown(response.text)
                    
                    # Extract JSON block and generate downloadable annotated PDF file
                    json_match = re.search(r"\[\s*\{.*\}\s*\]", response.text, re.DOTALL)
                    if json_match and student_file.type == "application/pdf":
                        json_str = json_match.group(0)
                        annotated_pdf_bytes = burn_annotations_to_pdf(student_bytes, json_str)
                        
                        st.download_button(
                            label="📥 Download Marked Student PDF",
                            data=annotated_pdf_bytes,
                            file_name=f"Marked_{student_file.name}",
                            mime="application/pdf"
                        )
                else:
                    st.error("All AI model servers are currently busy. Please try clicking the button again in 1 minute.")
                
            except Exception as e:
                st.error(f"Error processing assessment: {str(e)}")
