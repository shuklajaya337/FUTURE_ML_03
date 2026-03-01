# Resume Screening and Ranking System (ML-Powered)

An automated tool to screen and rank resumes based on their fit for a specific job description. This system extracts skills using NLP and calculates a weighted score based on text similarity and skill matching.

## Features
- **Text Cleaning**: Sanitizes resumes by removing URLs, hashtags, mentions, and special characters.
- **Skill Extraction**: Uses `spaCy`'s Matcher to extract technical skills automatically.
- **Similarity Scoring**: Calculates TF-IDF Cosine Similarity between resumes and the Job Description.
- **Weighted Ranking**: Combines skill matching (60%) and text similarity (40%) for a robust final rank.
- **Skill Gap Identification**: Highlights exact skills the candidate is missing.
- **Batch Processing**: Supports loading and ranking hundreds of resumes via `resumes.csv`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shuklajaya337/FUTURE_ML_03.git
   cd FUTURE_ML_03
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the spaCy Language Model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### 1. Command Line (Batch Processing)
- Prepare a `resumes.csv` with columns `Candidate_Name` and `Resume_Text`.
- Run:
  ```bash
  python ranker.py
  ```

### 2. Interactive Analysis
- Launch Jupyter Notebook:
  ```bash
  jupyter notebook
  ```
- Open `Resume_Screening_System.ipynb` to visualize the rankings and perform interactive screenings.

## Project Structure
- `nlp_module.py`: Text cleaning and NLP logic.
- `ranker.py`: Ranking algorithm and scoring system.
- `Resume_Screening_System.ipynb`: Main interactive data analysis.
- `resumes.csv`: Sample dataset template.

---
Built for the Future Intern ML Project.
