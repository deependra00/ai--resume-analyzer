import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# Page Layout Configuration
st.set_page_config(page_title="AI Resume & Skill Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI-Powered Resume & Skill Analyzer (Google Gemini)")
st.write("Upload your resume, paste a job description, and get instant ML scores + Gemini AI feedback!")

# Pre-defined Skill Database for Keyword Matching
SKILL_DATABASE = [
    "python", "java", "c++", "javascript", "typescript", "html", "css", "sql", "postgresql", "mongodb",
    "react", "node.js", "express", "angular", "vue", "django", "flask", "fastapi",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "keras", "pytorch",
    "scikit-learn", "pandas", "numpy", "power bi", "tableau", "excel", "data analytics", "data visualization",
    "git", "github", "docker", "kubernetes", "aws", "azure", "gcp", "linux", "dbeaver", "vscode",
    "communication", "leadership", "problem solving", "teamwork", "time management", "analytical skills"
]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

# Function to extract skills using regex database
def extract_skills(text):
    text_clean = text.lower()
    found_skills = set()
    for skill in SKILL_DATABASE:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_clean):
            found_skills.add(skill.capitalize())
    return found_skills

# Function to get AI Review from Gemini
def get_gemini_feedback(api_key, resume_text, job_description):
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        You are an expert technical recruiter and AI career coach.
        Analyze the following Resume and Job Description. Provide concise, professional feedback:
        1. Top 3 strengths of this candidate for this role.
        2. Top 3 missing areas or weaknesses.
        3. 2-3 specific actionable suggestions to improve the resume for this job description.
        
        Resume Text:
        {resume_text[:3000]}
        
        Job Description:
        {job_description[:1500]}
        """
        response = client.models.generate_content(
            model='gemini-flash-latest',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"

# API Key Input right on the Main Screen
st.subheader("🔑 Step 1: Enter Google Gemini API Key")
api_key_input = st.text_input("Paste your Gemini API Key here", type="password")

st.markdown("---")

# Layout UI - Two Columns for Resume & JD
col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

with col2:
    st.subheader("3. Job Description")
    job_description = st.text_area("Paste Job Description (JD) here", height=180)

# Main Execution Button
if st.button("🚀 Analyze with AI & ML", type="primary"):
    if api_key_input.strip() == "":
        st.error("⚠️ Please enter your Gemini API Key first!")
    elif uploaded_file is not None and job_description.strip() != "":
        with st.spinner("Processing text, computing similarity, and calling Gemini AI..."):
            # Step A: Extract Text
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if not resume_text.strip():
                st.error("Text extract nahi ho paaya. Kripya readable PDF upload karein.")
            else:
                # Step B: TF-IDF Similarity Score
                documents = [resume_text, job_description]
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(documents)
                match_score = cosine_similarity(tfidf_matrix)[0][1] * 100
                
                # Step C: Extract Skills
                resume_skills = extract_skills(resume_text)
                jd_skills = extract_skills(job_description)
                
                matched_skills = resume_skills.intersection(jd_skills)
                missing_skills = jd_skills - resume_skills
                
                # Step D: Results Section Display
                st.markdown("---")
                st.subheader("📊 Match Score & Skill Breakdown")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="Overall Match Score", value=f"{round(match_score, 2)}%")
                with res_col2:
                    if match_score >= 65:
                        st.success("🔥 Excellent Match for this role!")
                    elif match_score >= 40:
                        st.warning("⚠️ Moderate Match. Optimization Recommended.")
                    else:
                        st.error("❌ Low Match. Crucial skills missing.")
                
                st.progress(min(int(match_score), 100))
                
                # Skills Display
                skill_col1, skill_col2 = st.columns(2)
                with skill_col1:
                    st.subheader("✅ Matched Skills")
                    if matched_skills:
                        for skill in matched_skills:
                            st.write(f"- ✔️ **{skill}**")
                    else:
                        st.info("No explicit predefined skills matched.")
                        
                with skill_col2:
                    st.subheader("⚠️ Missing Skills")
                    if missing_skills:
                        for skill in missing_skills:
                            st.write(f"- ❌ **{skill}**")
                    else:
                        st.success("No critical predefined skills missing!")
                
                # Step E: Gemini AI Feedback Section
                st.markdown("---")
                st.subheader("🤖 Google Gemini AI Professional Feedback")
                with st.spinner("Asking Gemini AI for deep insights..."):
                    ai_feedback = get_gemini_feedback(api_key_input, resume_text, job_description)
                    st.markdown(ai_feedback)

    else:
        st.warning("⚠️ Please upload a PDF resume AND paste a valid Job Description.")