import streamlit as st
from google import genai
from google.genai import types

# 1. Configure Streamlit Page
st.set_page_config(page_title="Academic Marking Agent", layout="wide")
st.title("🎓 AI Academic Marking Agent")
st.markdown("Upload a **Student Answer Script** and a **Marking Scheme** to get visual score annotations and an Excel-ready summary matrix.")

# 2. Fetch API Key Securely from Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets. Please configure it in your Streamlit Cloud settings.")
    st.stop()

# 3. System Instructions Definition
SYSTEM_INSTRUCTIONS = """
You are a precise Academic Data Formatting Agent. 

You will analyze two documents uploaded by the user: a handwritten or typed Student Answer Script, and a master Marking Scheme. Your goal is to map out exact scores per question and output structured data meanted for spreadsheet population.

CRITICAL OPERATIONAL RULES:
- Do NOT rewrite or replicate any drawings, images, math equations, or text from the student's script. 
- Track student responses solely by their structural identifier (e.g., Q1, Q2, Q3a).
- Read diagrams (such as reaction mechanisms) dynamically. Evaluate if mandatory features (arrows, intermediates, valencies, labels) match the marking scheme, decide the grade, but communicate this strictly as a numerical deduction and a structural textual comment.

OUTPUT REQUIREMENTS:

### SECTION 1: VISUAL ANNOTATION LIST
Produce a short list detailing what marks to physically draw over each question on the student's paper. Use this exact syntax:
- [Question Number]: Write "[Score Awarded] / [Max Score]"

### SECTION 2: EXCEL SPREADSHEET MATRIX
Provide a clean Markdown table containing your granular evaluations. Format your table with these exact columns:
| Question # | Max Marks | Marks Awarded | Deductions | Deduction Reason / Feedback Summary |
"""

# 4. Multimodal File Upload UI
col1, col2 = st.columns(2)
with col1:
    student_file = st.file_uploader("Upload Student Script (PDF / Image)", type=["pdf", "png", "jpg", "jpeg"])
with col2:
    scheme_file = st.file_uploader("Upload Marking Scheme (PDF / Image)", type=["pdf", "png", "jpg", "jpeg"])

# 5. Assessment Execution Logic
if st.button("🚀 Run Assessment Marking"):
    if not student_file or not scheme_file:
        st.warning("Please upload both the Student Answer Script and the Marking Scheme.")
    else:
        with st.spinner("Analyzing handwritten diagrams and generating score matrices..."):
            try:
                # Initialize GenAI Client securely with secret key
                client = genai.Client(api_key=api_key)
                
                # Format files for multimodal API call
                student_bytes = student_file.read()
                scheme_bytes = scheme_file.read()
                
                parts = [
                    types.Part.from_bytes(data=student_bytes, mime_type=student_file.type),
                    types.Part.from_bytes(data=scheme_bytes, mime_type=scheme_file.type),
                    "Grade the student script against the marking scheme according to system instructions."
                ]
                
                # Execute model call using Gemini Flash
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTIONS
                    )
                )
                
                st.success("Marking Complete!")
                st.markdown("---")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Error processing assessment: {str(e)}")
