import streamlit as st
import pandas as pd
import pdfplumber
import io
from ranker import ResumeRanker
import base64

# Page Config
st.set_page_config(
    page_title="Resume Screening & Ranking System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        border: none;
        color: white;
    }
    .score-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #10B981;
    }
    .skill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        background-color: #E0E7FF;
        color: #3730A3;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .missing-badge {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Initialize Ranker
@st.cache_resource
def get_ranker():
    return ResumeRanker()

ranker = get_ranker()

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    st.info("Upload resumes and enter the Job Description to find the best candidates.")
    
    st.divider()
    
    st.markdown("### 🛠️ About the System")
    st.write("This AI-powered tool uses NLP (TF-IDF & spaCy) to rank candidates based on skill matching and text similarity.")

# Main Header
st.title("📄 Resume Screening & Ranking System")
st.markdown("---")

# Layout: Two columns for input
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Job Description")
    jd_input = st.text_area("Paste the Job Description here:", height=300, placeholder="e.g., We are looking for a Data Scientist with experience in Python, SQL, and Machine Learning...")

with col2:
    st.subheader("📤 Upload Resumes")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    
    st.write("OR")
    
    use_sample = st.checkbox("Use sample resumes from CSV", value=False)
    if use_sample:
        st.info("Loading sample candidates from 'resumes.csv'...")

# Ranking Logic
if st.button("🚀 Rank Candidates"):
    if not jd_input:
        st.error("Please enter a Job Description.")
    elif not uploaded_files and not use_sample:
        st.error("Please upload resumes or use sample data.")
    else:
        with st.spinner("Analyzing resumes and calculating scores..."):
            # Prepare DataFrame
            resumes_list = []
            
            # 1. Process Uploaded PDFs
            if uploaded_files:
                for file in uploaded_files:
                    try:
                        text = extract_text_from_pdf(file)
                        resumes_list.append({
                            'Candidate_Name': file.name.replace(".pdf", ""),
                            'Resume_Text': text
                        })
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
            
            # 2. Process Sample CSV
            if use_sample:
                try:
                    df_sample = pd.read_csv("resumes.csv")
                    resumes_list.extend(df_sample.to_dict('records'))
                except Exception as e:
                    st.error(f"Error loading resumes.csv: {e}")
            
            if resumes_list:
                df_resumes = pd.DataFrame(resumes_list)
                
                # Perform Ranking
                results_df = ranker.rank_candidates(jd_input, df_resumes)
                
                # Show Results
                st.markdown("---")
                st.header("🏆 Best Matches")
                
                # Display Top Candidate as a Highlight
                top_candidate = results_df.iloc[0]
                st.success(f"### 🥇 Top Pick: **{top_candidate['Candidate_Name']}** (Score: {top_candidate['Final_Score']:.2f})")
                
                # Detailed Results Table
                st.subheader("📊 Detailed Rankings")
                
                # Create a cleaner display DF
                display_df = results_df[['Candidate_Name', 'Final_Score', 'Skill_Match_Score', 'Similarity_Score']].copy()
                display_df.columns = ['Candidate Name', 'Internal Score', 'Skill Match', 'Text Similarity']
                
                st.dataframe(display_df.style.background_gradient(cmap='Greens', subset=['Internal Score']), use_container_width=True)
                
                # Interactive Detailed Analysis
                st.subheader("🔍 Skill Gap Analysis")
                selected_candidate = st.selectbox("Select a candidate to view detailed analysis:", results_df['Candidate_Name'].tolist())
                
                candidate_data = results_df[results_df['Candidate_Name'] == selected_candidate].iloc[0]
                
                c_col1, c_col2 = st.columns(2)
                
                with c_col1:
                    st.markdown("#### ✅ Matched Skills")
                    if candidate_data['Matched_Skills']:
                        html = "".join([f'<span class="skill-badge">{s}</span>' for s in candidate_data['Matched_Skills']])
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.write("No direct skill matches found.")
                        
                with c_col2:
                    st.markdown("#### ❌ Missing Skills")
                    if candidate_data['Missing_Skills']:
                        html = "".join([f'<span class="skill-badge missing-badge">{s}</span>' for s in candidate_data['Missing_Skills']])
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.write("No missing skills identified! (Perfect match)")
                
                st.markdown("---")
                # Option to download results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Rankings as CSV",
                    data=csv,
                    file_name='resume_rankings.csv',
                    mime='text/csv',
                )
            else:
                st.warning("No valid resume text found to analyze.")

# Footer
st.markdown("---")
st.markdown("<center>Built for Future Intern ML Project | Powered by Streamlit & spaCy</center>", unsafe_allow_html=True)
