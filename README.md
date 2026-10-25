# AI-Powered Resume Analyzer & ATS Matcher

An intelligent Full-Stack AI application designed to evaluate resumes against job descriptions using Google Gemini API. This tool helps job seekers optimize their resumes, calculate match percentages, and identify missing keywords to clear Applicant Tracking Systems (ATS).

---

## Tech Stack & Tools
* **AI & NLP:** Google Gemini API, Python
* **Backend:** Python (`app.py`), Flask / FastAPI
* **Libraries:** `pypdf`, `google-generativeai`, `python-dotenv`
* **Version Control:** Git & GitHub

---

## Key Features
* **PDF Resume Parsing:** Extracts clean text seamlessly from uploaded PDF resumes.
* **Smart ATS Scoring:** Compares resume content with Job Descriptions (JD) to provide a precise match score percentage.
* **AI-Driven Insights:** Powered by Google Gemini to suggest specific improvements, missing skills, and formatting recommendations.
* **User-Friendly Interface:** Clean and intuitive workflow for instant analysis.

---

## Project Structure
```text
ai-resume-analyzer/
│
├── app.py              # Main Python application & Gemini integration
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
