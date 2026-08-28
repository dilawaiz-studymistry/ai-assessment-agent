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

# 3. Enhanced System Instructions with Strict Sub-Question Rules
SYSTEM_INSTRUCTIONS = """
You are a precise Academic Data Formatting Agent. 

You will analyze two documents uploaded by the user: a handwritten or typed Student Answer Script, and a master Marking Scheme. Your goal is to map out exact scores per question and output structured data meant for spreadsheet population.

CRITICAL OPERATIONAL RULES:
- Do NOT rewrite or replicate any drawings, images, math equations, or text from the student's script. 
- Track student responses solely by their structural identifier (e.g., Q1, Q2, Q4(b)i, Q4(b)ii).
- Read diagrams (such as reaction mechanisms) dynamically. Evaluate if mandatory features match the scheme, decide the grade, and communicate this strictly as a numerical deduction and structural feedback.

OUTPUT REQUIREMENTS:

SECTION 1: VISUAL ANNOTATION LIST
Produce a short list detailing what marks to physically draw over each question on the student's paper. Use this exact syntax:
- [Question Number]: Write "[Score Awarded] / [Max Score]"

SECTION 2: EXCEL SPREADSHEET MATRIX
Provide a clean Markdown table containing your granular evaluations. Format your table with these exact columns:
| Question # | Max Marks | Marks Awarded | Deductions | Deduction Reason / Feedback Summary |

SECTION 3: JSON STRUCTURE FOR PDF STAMPING
AT THE VERY END OF YOUR RESPONSE, provide a raw JSON array. For each evaluated item (including all sub-parts), provide:
- 'page': page index (1-based)
- 'target_text': exact full label string as printed on the student paper for matching (e.g. "4(b)i", "4(b)ii", "(ii)", "Q4b"). NEVER use isolated numbers like "ii" without context.
- 'score_text': score summary text to stamp (e.g. "4(b)ii: 2/2")

Example JSON structure required at the very end:
[
  {
    "page": 1,
    "target_text": "4(b)i",
    "score_text": "4(b)i: 1/1"
  },
  {
    "page": 1,
    "target_text": "4(b)ii",
    "score_text": "4(b)ii: 2/2"
  }
]
"""

# 4. Helper Function: Disambiguated Search + Collision Avoidance
def burn_annotations_to_pdf(pdf_bytes, json_data):
    """
    Locates question/sub-question headings programmatically on PDF pages,
    places score stamps in the right margin, and prevents overlaps.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    try:
        annotations = json.loads(json_data)
        
        # Track used vertical slots per page to prevent overlap collisions
        used_y_coords = {}
        
        for item in annotations:
            page_num = item.get("page", 1) - 1  # 0-indexed page index
            if 0 <= page_num < len(doc):
                page = doc[page_num]
                target_text = str(item.get("target_text", "")).strip()
                score_text = str(item.get("score_text", "")).strip()
                
                page_width = page.rect.width
                page_height = page.rect.height
                
                # Dynamic right margin placement
                x = max(page_width - 170.0, 50.0)
                found_y = None
                
                # Multi-stage Search Strategy for Sub-questions
                if target_text:
                    # 1. Exact Search
                    matches = page.search_for(target_text)
                    
                    # 2. Sub-part fallback: search for "(ii)" if "4(b)ii" wasn't found as a single block
                    if not matches and "(" in target_text and ")" in target_text:
                        sub_part = target_text[target_text.find("("):]
                        matches = page.search_for(sub_part)
                        
                    if matches:
                        # Grab top-most match for this sub-question header
                        found_y = matches[0].y1 + 2.0

                # Fallback location if PyMuPDF couldn't locate target_text
                if found_y is None:
                    found_y = 100.0
                
                # ANTI-COLLISION CHECK: If another stamp is already within 18px of this location, shift downward
                if page_num not in used_y_coords:
                    used_y_coords[page_num] = []
                
                for existing_y in used_y_coords[page_num]:
                    if abs(existing_y - found_y) < 18.0:
                        found_y = existing_y + 18.0  # Push down by line-height step
                
                used_y_coords[page_num].append(found_y)
                
                # Clamp Y inside printable limits
                y = min(max(found_y, 25.0), page_height - 25.0)
                
                # Render red score text
                try:
                    page.insert_text(
                        fitz.Point(x, y),
                        score_text,
                        fontsize=11,
                        color=(0.85, 0, 0),  # Red
                        fontname="helv"
                    )
                except Exception:
                    rect = fitz.Rect(x, y, x + 150, y + 20)
                    annot = page.add_freetext_annot(rect, score_text, fontsize=11, text_color=(0.85, 0, 0))
                    annot.update()
                    
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
                client = genai.Client(api_key=api_key)
                
                student_bytes = student_file.read()
                scheme_bytes = scheme_file.read()
                
                parts = [
                    types.Part.from_bytes(data=student_bytes, mime_type=student_file.type),
                    types.Part.from_bytes(data=scheme_bytes, mime_type=scheme_file.type),
                    "Grade the student script against the marking scheme according to system instructions."
                ]
                
                models_to_try = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
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
                    except Exception:
                        continue

                if response:
                    st.success("Marking Complete!")
                    st.markdown("---")
                    
                    st.markdown(response.text)
                    
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
                    st.error("Both models were unavailable. Please try again in 1 minute.")
                
            except Exception as e:
                st.error(f"Error processing assessment: {str(e)}")
