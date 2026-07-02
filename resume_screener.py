"""
AI-Based Resume Screening System
---------------------------------
Ranks candidate resumes against a job description using TF-IDF embeddings
and cosine similarity, with keyword-level explainability.

Pipeline:
    1. Extract text from resumes (.txt or .pdf) and the job description
    2. Preprocess text (lowercase, clean)
    3. Generate TF-IDF vector embeddings
    4. Compute cosine similarity between each resume and the job description
    5. Rank candidates and show top matching keywords (explainability)

Note: This uses TF-IDF (scikit-learn) instead of transformer embeddings so it
runs fully offline with no model downloads. To upgrade to transformer-based
embeddings later, install `sentence-transformers` and swap the
`build_tfidf_vectors()` function for a call to
SentenceTransformer('all-MiniLM-L6-v2').encode(texts) -- the rest of the
ranking/explainability logic stays the same.
"""

import os
import re
import glob
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- 1. Text extraction ----------

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(path):
    if path.lower().endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif path.lower().endswith(".txt"):
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def load_resumes(folder):
    """Loads all .txt and .pdf resumes from a folder."""
    resumes = {}
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if path.lower().endswith((".txt", ".pdf")):
            name = os.path.splitext(os.path.basename(path))[0]
            resumes[name] = load_document(path)
    return resumes


# ---------- 2. Preprocessing ----------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- 3. Embeddings (TF-IDF) ----------

def build_tfidf_vectors(job_description, resume_texts):
    """
    Fits a TF-IDF vectorizer on [job_description + all resumes] and returns
    the vectorizer plus the resulting matrix. Index 0 is always the JD.
    """
    documents = [job_description] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


# ---------- 4. Similarity & ranking ----------

def rank_candidates(job_description, resumes: dict):
    names = list(resumes.keys())
    cleaned_resumes = [clean_text(text) for text in resumes.values()]
    cleaned_jd = clean_text(job_description)

    vectorizer, tfidf_matrix = build_tfidf_vectors(cleaned_jd, cleaned_resumes)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(jd_vector, resume_vectors).flatten()

    results = []
    feature_names = vectorizer.get_feature_names_out()

    for i, name in enumerate(names):
        top_keywords = get_top_matching_keywords(
            jd_vector, resume_vectors[i], feature_names, top_n=6
        )
        results.append({
            "candidate": name,
            "match_score": round(float(scores[i]) * 100, 2),
            "top_matching_keywords": ", ".join(top_keywords)
        })

    df = pd.DataFrame(results).sort_values(by="match_score", ascending=False)
    df.reset_index(drop=True, inplace=True)
    df.index += 1  # rank starts at 1
    df.index.name = "rank"
    return df


# ---------- 5. Explainability ----------

def get_top_matching_keywords(jd_vector, resume_vector, feature_names, top_n=6):
    """
    Finds words that are highly weighted in BOTH the job description and the
    resume -- i.e. the keywords actually driving the similarity score.
    """
    jd_arr = jd_vector.toarray().flatten()
    resume_arr = resume_vector.toarray().flatten()

    overlap_scores = jd_arr * resume_arr  # elementwise product
    top_indices = overlap_scores.argsort()[::-1][:top_n]
    top_keywords = [feature_names[i] for i in top_indices if overlap_scores[i] > 0]
    return top_keywords


# ---------- Main ----------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jd_path = os.path.join(base_dir, "data", "job_description.txt")
    resumes_dir = os.path.join(base_dir, "data", "resumes")

    job_description = load_document(jd_path)
    resumes = load_resumes(resumes_dir)

    print(f"Loaded {len(resumes)} resumes.")
    print("Ranking candidates against job description...\n")

    ranked_df = rank_candidates(job_description, resumes)

    pd.set_option("display.max_colwidth", 60)
    print(ranked_df.to_string())

    output_path = os.path.join(base_dir, "ranked_candidates.csv")
    ranked_df.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
