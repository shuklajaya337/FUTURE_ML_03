import re
import spacy
from spacy.matcher import Matcher

class ResumeNLP:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback if model not found
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_skill_matcher()

    def clean_text(self, text):
        """Clean resume/JD text: remove URLs, special chars, extra whitespace."""
        text = re.sub(r'http\S+\s*', ' ', text)  # remove URLs
        text = re.sub(r'RT|cc', ' ', text)  # remove RT and cc
        text = re.sub(r'#\S+', '', text)  # remove hashtags
        text = re.sub(r'@\S+', '  ', text)  # remove mentions
        text = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
        text = re.sub(r'[^\x00-\x7f]', r' ', text) 
        text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
        return text.strip().lower()

    def _setup_skill_matcher(self):
        """Define patterns for skill extraction."""
        # This is a basic list; in a real app, this would be a large JSON or DB
        skills = [
            'Python', 'Java', 'C++', 'SQL', 'Machine Learning', 'Deep Learning',
            'NLP', 'Data Science', 'React', 'Angular', 'Node.js', 'AWS', 'Azure',
            'Docker', 'Kubernetes', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow',
            'PyTorch', 'Tableau', 'Power BI', 'Excel', 'C#', 'Javascript', 'HTML', 'CSS'
        ]
        
        for skill in skills:
            pattern = [{"LOWER": s.lower()} for s in skill.split()]
            self.matcher.add(skill, [pattern])

    def extract_skills(self, text):
        """Extract skills from text using spaCy Matcher."""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        found_skills = set()
        for match_id, start, end in matches:
            found_skills.add(self.nlp.vocab.strings[match_id])
        return list(found_skills)

if __name__ == "__main__":
    nlp_engine = ResumeNLP()
    sample_text = "Experienced Data Scientist with skills in Python, Machine Learning, and SQL. Familiar with AWS."
    cleaned = nlp_engine.clean_text(sample_text)
    print(f"Cleaned: {cleaned}")
    print(f"Skills: {nlp_engine.extract_skills(cleaned)}")
