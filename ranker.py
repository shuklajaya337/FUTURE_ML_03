import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp_module import ResumeNLP

class ResumeRanker:
    def __init__(self):
        self.nlp = ResumeNLP()
        self.vectorizer = TfidfVectorizer()

    def rank_candidates(self, job_description, resumes_df):
        """
        Rank resumes based on similarity to JD and skill matching.
        resumes_df should have columns: 'Resume_Text', 'Candidate_Name'
        """
        # Clean JD
        clean_jd = self.nlp.clean_text(job_description)
        jd_skills = set(self.nlp.extract_skills(clean_jd))
        
        # Clean Resumes
        resumes_df['Cleaned_Resume'] = resumes_df['Resume_Text'].apply(self.nlp.clean_text)
        
        # 1. Cosine Similarity using TF-IDF
        all_text = [clean_jd] + resumes_df['Cleaned_Resume'].tolist()
        tfidf_matrix = self.vectorizer.fit_transform(all_text)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # 2. Skill Matching & Gap Identification
        def evaluate_skills(resume_text):
            res_skills = set(self.nlp.extract_skills(resume_text))
            matched = jd_skills.intersection(res_skills)
            missing = jd_skills - res_skills
            match_score = len(matched) / len(jd_skills) if jd_skills else 0
            return matched, missing, match_score

        results = resumes_df['Cleaned_Resume'].apply(evaluate_skills)
        resumes_df['Matched_Skills'] = results.apply(lambda x: list(x[0]))
        resumes_df['Missing_Skills'] = results.apply(lambda x: list(x[1]))
        resumes_df['Skill_Match_Score'] = results.apply(lambda x: x[2])
        resumes_df['Similarity_Score'] = cosine_sim
        
        # 3. Final Weighted Score (60% Skill Match, 40% Text Similarity)
        resumes_df['Final_Score'] = (resumes_df['Skill_Match_Score'] * 0.6) + (resumes_df['Similarity_Score'] * 0.4)
        
        # Rank and sort
        return resumes_df.sort_values(by='Final_Score', ascending=False)

if __name__ == "__main__":
    import os

    # 1. Load Data
    csv_file = "resumes.csv"
    if os.path.exists(csv_file):
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        # Ensure required columns exist
        if 'Resume_Text' not in df.columns or 'Candidate_Name' not in df.columns:
            print("Error: CSV must have 'Candidate_Name' and 'Resume_Text' columns.")
            exit(1)
    else:
        print("CSV file not found. Using sample test data...")
        data = {
            'Candidate_Name': ['Alice', 'Bob', 'Charlie'],
            'Resume_Text': [
                "Data Scientist with Python, SQL, and Machine Learning experience.",
                "Java Developer with Spring Boot and SQL knowledge.",
                "Python Developer specializing in NLP and AWS cloud."
            ]
        }
        df = pd.DataFrame(data)

    # 2. Input Job Description
    jd = "Seeking a Data Scientist proficient in Python, SQL, and Machine Learning. AWS is a plus."
    
    # 3. Rank
    ranker = ResumeRanker()
    ranked_df = ranker.rank_candidates(jd, df)
    
    # 4. Show Results (Limited to top 10 for readability)
    print("\n--- Final Rankings ---")
    print(ranked_df[['Candidate_Name', 'Final_Score', 'Matched_Skills', 'Missing_Skills']].head(10))
