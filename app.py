import streamlit as st
import pandas as pd
import pdfplumber
import io
import os
import kagglehub
import plotly.express as px
import plotly.graph_objects as go
from ranker import ResumeRanker
import base64

# Page Config
st.set_page_config(
    page_title="Pro Resume Screener & Ranker",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f5;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    .sidebar-content {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

@st.cache_resource
def get_ranker():
    return ResumeRanker()

@st.cache_data
def load_kaggle_data():
    try:
        path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
        csv_path = os.path.join(path, "Resume", "Resume.csv")
        df = pd.read_csv(csv_path)
        # Map columns to match our expectation
        df = df.rename(columns={'Resume_str': 'Resume_Text', 'ID': 'Candidate_Name'})
        df['Candidate_Name'] = df['Candidate_Name'].astype(str) + " (" + df['Category'] + ")"
        return df
    except Exception as e:
        st.error(f"Failed to load Kaggle dataset: {e}")
        return None

ranker = get_ranker()

# Sidebar
with st.sidebar:
    st.title("🚀 Pro Settings")
    
    st.subheader("⚖️ Weighting Logic")
    skill_weight = st.slider("Skill Match Weight (%)", 0, 100, 60)
    sim_weight = 100 - skill_weight
    st.caption(f"Current split: {skill_weight}% Skills / {sim_weight}% General Similarity")
    
    st.divider()
    
    st.subheader("📦 Dataset Source")
    data_source = st.radio("Choose Resume Source:", ["Manual Upload", "Local Sample (CSV)", "Kaggle Dataset (2400+ Resumes)"])
    
    if data_source == "Kaggle Dataset (2400+ Resumes)":
        num_resumes = st.slider("Number of resumes to pull from Kaggle", 10, 500, 100)
    
    st.divider()
    st.markdown("### 📊 System Analytics")
    st.info("The system uses spaCy for Named Entity Recognition and TF-IDF for semantic similarity.")

# Main Header
st.title("📄 Pro Resume Screening & Ranking System")
st.markdown("---")

# Layout: Inputs
col_jd, col_files = st.columns([1, 1])

with col_jd:
    st.header("📝 Job Description")
    jd_input = st.text_area("Paste the Job Description here:", height=250, placeholder="e.g., Senior Python Developer with 5+ years experience in Flask and AWS...")

with col_files:
    st.header("📥 Resume Input")
    if data_source == "Manual Upload":
        uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    elif data_source == "Local Sample (CSV)":
        st.info("Loading from `resumes.csv`...")
        uploaded_files = None
    else:
        st.success(f"Kaggle Dataset is ready! (Pulling top {num_resumes} records)")
        uploaded_files = None

# Ranking Logic
if st.button("🚀 Run Advanced Ranking"):
    if not jd_input:
        st.error("Please enter a job description.")
    else:
        with st.spinner("Analyzing and calculating weights..."):
            # Data Preparation
            resumes_list = []
            
            if data_source == "Manual Upload" and uploaded_files:
                for file in uploaded_files:
                    try:
                        text = extract_text_from_pdf(file)
                        resumes_list.append({'Candidate_Name': file.name.replace(".pdf", ""), 'Resume_Text': text})
                    except: pass
            elif data_source == "Local Sample (CSV)":
                try:
                    df_sample = pd.read_csv("resumes.csv")
                    resumes_list = df_sample.to_dict('records')
                except: pass
            elif data_source == "Kaggle Dataset (2400+ Resumes)":
                df_kaggle = load_kaggle_data()
                if df_kaggle is not None:
                    resumes_list = df_kaggle.head(num_resumes).to_dict('records')

            if resumes_list:
                df_resumes = pd.DataFrame(resumes_list)
                
                # Perform Ranking with Dynamic Weights
                # Temporarily override the weights in the ranker logic if possible, 
                # but since ranker.py is static, we'll re-calculate the final score here
                results_df = ranker.rank_candidates(jd_input, df_resumes)
                
                # Recalculate with dynamic weights
                results_df['Final_Score'] = (results_df['Skill_Match_Score'] * (skill_weight/100)) + \
                                           (results_df['Similarity_Score'] * (sim_weight/100))
                results_df = results_df.sort_values(by='Final_Score', ascending=False)

                # --- DASHBOARD ---
                st.markdown("---")
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Candidates Processed", len(results_df))
                m2.metric("Average Score", f"{results_df['Final_Score'].mean():.2f}")
                m3.metric("Top Score", f"{results_df['Final_Score'].max():.2f}")

                # Charts Row
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.subheader("📊 Ranking Distribution")
                    fig = px.bar(results_df.head(15), x='Candidate_Name', y='Final_Score', 
                                 color='Final_Score', color_continuous_scale='Viridis',
                                 labels={'Candidate_Name': 'Candidate', 'Final_Score': 'Overall Fit'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.subheader("🎯 Average Skill Fit")
                    # Show a donut chart of average skill match vs missing
                    avg_match = results_df['Skill_Match_Score'].mean()
                    fig_donut = go.Figure(data=[go.Pie(labels=['Matched', 'Gaps'], 
                                                      values=[avg_match, 1-avg_match], 
                                                      hole=.6,
                                                      marker_colors=['#10B981', '#EF4444'])])
                    fig_donut.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_donut, use_container_width=True)

                # Results Table
                st.divider()
                st.subheader("🏆 Detailed Ranking Table")
                st.dataframe(results_df[['Candidate_Name', 'Final_Score', 'Skill_Match_Score', 'Similarity_Score']]
                             .style.background_gradient(cmap='YlGnBu', subset=['Final_Score']), 
                             use_container_width=True)

                # Deep Dive
                st.divider()
                st.header("🔍 Individual Candidate Deep-Dive")
                selected_name = st.selectbox("Select candidate to inspect:", results_df['Candidate_Name'].tolist())
                cand = results_df[results_df['Candidate_Name'] == selected_name].iloc[0]

                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.markdown("#### ✅ Skills Found")
                    if cand['Matched_Skills']:
                        st.markdown("".join([f'<span class="skill-badge">{s}</span>' for s in cand['Matched_Skills']]), unsafe_allow_html=True)
                    else: st.write("None")
                
                with d_col2:
                    st.markdown("#### ❌ Skill Gaps")
                    if cand['Missing_Skills']:
                        st.markdown("".join([f'<span class="skill-badge missing-badge">{s}</span>' for s in cand['Missing_Skills']]), unsafe_allow_html=True)
                    else: st.success("Perfect Match!")

                st.markdown("#### 📝 Cleaned Resume Excerpt")
                st.text_area("Preview:", value=cand['Cleaned_Resume'][:1000] + "...", height=150, disabled=True)

            else:
                st.warning("No resumes found to analyze. Check your upload or dataset connection.")

# Footer
st.markdown("---")
st.markdown("<center>Resume AI Pro | Future Intern ML Track | Created by Antigravity</center>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center>Built for Future Intern ML Project | Powered by Streamlit & spaCy</center>", unsafe_allow_html=True)
