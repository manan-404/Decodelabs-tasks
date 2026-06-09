# Project 3: Career Path Recommendation System
# Method: TF-IDF Vectorization + Cosine Similarity

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── 1. LOAD DATASET ─────────────────────────────────────────────────────────

df = pd.read_csv("raw_skills.csv")

print("=" * 55)
print(" CAREER PATH RECOMMENDATION SYSTEM")
print("=" * 55)
print(f"Dataset loaded: {len(df)} job roles found")
print(f"Roles available: {', '.join(df['job_role'].tolist())}\n")

# ─── 2. USER INPUT ───────────────────────────────────────────────────────────

print("Enter at least 3 skills to get your top career matches.")
print("Example: python, machine learning, docker\n")

user_skills = []

while len(user_skills) < 3:
    remaining = 3 - len(user_skills)
    skill = input(f"Enter skill {len(user_skills) + 1}: ").lower().strip()

    if not skill:
        print("  Skill cannot be empty. Try again.")
        continue

    user_skills.append(skill)

    if len(user_skills) < 3:
        print(f"  ({remaining - 1} more required)")

print("\nWant to add more skills? (press Enter to skip)")
while True:
    extra = input(f"Enter skill {len(user_skills) + 1} (or press Enter to continue): ").lower().strip()
    if not extra:
        break
    user_skills.append(extra)

user_input_string = " ".join(user_skills)

print(f"\nYour skills : {', '.join(user_skills)}")

# ─── 3. TF-IDF VECTOR MAPPING ────────────────────────────────────────────────

all_documents = df["skills"].tolist() + [user_input_string]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_documents)

# User vector is the last row, job vectors are everything before it
job_vectors  = tfidf_matrix[:-1]
user_vector  = tfidf_matrix[-1]

print("\n" + "=" * 55)
print(" VECTOR MAPPING COMPLETE")
print("=" * 55)
print(f"Vocabulary size : {len(vectorizer.vocabulary_)} unique terms")
print(f"User vector     : {user_vector.nnz} non-zero features mapped")

# ─── 4. COSINE SIMILARITY SCORING ────────────────────────────────────────────

similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

df["similarity_score"] = similarity_scores

# ─── 5. SORT & FILTER — TOP 3 ────────────────────────────────────────────────

top3 = df.sort_values("similarity_score", ascending=False).head(3).reset_index(drop=True)

print("\n" + "=" * 55)
print(" TOP 3 CAREER PATH MATCHES")
print("=" * 55)

for i, row in top3.iterrows():
    rank  = i + 1
    role  = row["job_role"]
    score = row["similarity_score"]
    bar   = "█" * int(score * 30)

    print(f"\n  #{rank}  {role}")
    print(f"       Score : {score:.4f}  {bar}")

print("\n" + "=" * 55)
print(" ALL ROLES RANKED")
print("=" * 55)

all_ranked = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
for i, row in all_ranked.iterrows():
    print(f"  {i+1:>2}. {row['job_role']:<25} {row['similarity_score']:.4f}")

print()
