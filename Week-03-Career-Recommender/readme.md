# Week 03 — Career Path Recommendation System

A content-based recommendation engine that matches a user's skills to the
most relevant career paths using TF-IDF vectorization and Cosine Similarity.

## How to Run

Install dependencies:

    pip install scikit-learn pandas

Then run:

    python recommender.py

## Dataset

**raw_skills.csv** — a custom dataset of 15 job roles with associated skill sets.

| Job Role | Example Skills |
|----------|---------------|
| Data Scientist | python, machine learning, statistics, tensorflow |
| Cybersecurity Analyst | penetration testing, ethical hacking, cryptography |
| DevOps Engineer | docker, kubernetes, ci/cd, terraform |
| Blockchain Developer | solidity, ethereum, smart contracts, web3 |
| ... | ... |

Full dataset included in this folder.

## Pipeline

    User Input (3+ skills) → TF-IDF Vectorization → Cosine Similarity Scoring → Top 3 Results

## Concepts Demonstrated

- Content-based filtering using skill profile matching
- TF-IDF weighting to penalize generic keywords and reward specific skills
- Cosine Similarity scoring (chosen over Euclidean distance to avoid
  sensitivity to vector length)
- Sorting and filtering results to display a ranked Top-3 output
- Input validation loop enforcing a minimum of 3 skills

## Why Cosine Similarity?

Two job profiles — one with 10 skills and one with 3 — have very different
vector magnitudes. Euclidean distance would unfairly penalize the shorter
profile. Cosine Similarity ignores magnitude and only measures the angle
between vectors, making it the correct choice for text-based matching.

## Project Structure

    Week-03-Career-Recommender/
    ├── recommender.py
    ├── raw_skills.csv
    └── README.md

## Sample Output

    Enter skill 1: python
    Enter skill 2: machine learning
    Enter skill 3: docker

    TOP 3 CAREER PATH MATCHES

      #1  ML Engineer
           Score : 0.5823  █████████████████

      #2  Data Scientist
           Score : 0.4901  ██████████████

      #3  DevOps Engineer
           Score : 0.3102  █████████
