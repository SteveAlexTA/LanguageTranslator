import streamlit as st
from chatbot import rag_chain

st.set_page_config(
    page_title="IELTS Writing Task 2 Examiner",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Slate / Teal Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 100%);
        color: #e0e1dd;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Input Text Areas */
    .stTextArea textarea {
        background-color: #1b263b !important;
        color: #ffffff !important;
        border: 1px solid #415a77 !important;
        border-radius: 10px !important;
        font-size: 15px !important;
    }
    
    .stTextArea textarea:focus {
        border: 1px solid #00b4d8 !important;
        box-shadow: 0 0 8px rgba(0, 180, 216, 0.4) !important;
    }
    
    /* Primary Action Button */
    .stButton button {
        background: linear-gradient(90deg, #0077b6 0%, #00b4d8 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.4);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1b2a !important;
        border-right: 1px solid #1c2541 !important;
    }
    
    /* Report Card Container */
    .report-box {
        background-color: rgba(27, 38, 59, 0.7);
        border: 1px solid #415a77;
        border-radius: 12px;
        padding: 20px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Information
with st.sidebar:
    st.title("🎓 IELTS Scoring Guide")
    st.markdown("""
    **Evaluation Dimensions:**
    * **Task Achievement (TA):** Complete prompt response & clear stance.
    * **Coherence & Cohesion (CC):** Logical structure & cohesive links.
    * **Lexical Resource (LR):** Advanced vocabulary & academic collocations.
    * **Grammatical Range & Accuracy (GRA):** Complex clauses & syntax control.
    
    ---
    📊 **Standard:** 250+ Words Required
    """)

# Main Page Header
st.title("🎓 IELTS Writing Task 2 Examiner")
st.caption("Automated IELTS Scoring & Diagnostic Feedback powered by LLM RAG Benchmarks")

# Dual Column Submission Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Your Essay Submission")
    
    prompt_text = st.text_area(
        "IELTS Task 2 Prompt:",
        placeholder="e.g., Some people believe that unpaid community service should be compulsory in high schools...",
        height=110
    )
    
    essay_text = st.text_area(
        "Your Response:",
        placeholder="Paste your complete essay here (minimum 250 words)...",
        height=360
    )
    
    word_count = len(essay_text.split()) if essay_text.strip() else 0
    if 0 < word_count < 250:
        st.warning(f"Word count: **{word_count} / 250** words (Under-length penalty applies)")
    elif word_count >= 250:
        st.success(f"Word count: **{word_count}** words")
    else:
        st.caption("Word count: 0 words")
        
    submit_btn = st.button("Evaluate Essay", type="primary", use_container_width=True)

with col2:
    st.subheader("Examiner Report")
    
    if submit_btn:
        if not prompt_text.strip() or not essay_text.strip():
            st.warning("Please provide both the essay prompt and your written essay.")
        else:
            combined_query = f"IELTS Prompt: {prompt_text.strip()}\n\nCandidate Essay:\n{essay_text.strip()}"
            with st.spinner("Analyzing essay against official IELTS benchmarks..."):
                try:
                    # Stream tokens in real time directly to the UI
                    st.write_stream(rag_chain.stream(combined_query))
                except Exception as e:
                    st.error(f"Evaluation error: {e}")