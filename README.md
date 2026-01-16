# Netflix-Recommendation-Engine# 

Netflix Recommendation Engine using TF-IDF and SVD

A **Netflix-style movie recommendation system** that combines **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Approach** using **TF-IDF** and **Matrix Factorization (SVD)**.

---

## 📌 Features

- 🎭 **Content-Based Filtering**
  - Uses TF-IDF vectorization on movie genres
  - Computes similarity using cosine similarity
  - Recommends movies similar to a given title

- 👥 **Collaborative Filtering**
  - Uses **Singular Value Decomposition (SVD)**
  - Learns user–movie interaction patterns
  - Predicts unseen movie ratings for users

- 🔀 **Hybrid Recommendation System**
  - Combines content similarity and collaborative predictions
  - Adjustable weighting parameter (`alpha`)
  - Improves recommendation quality and personalization

---

## 🛠️ Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Scikit-Surprise  
- TF-IDF Vectorization  
- Cosine Similarity  
- Matrix Factorization (SVD)

---

## 📂 Dataset Requirements

This project uses the **MovieLens dataset**.

Required files:
- `movies.csv`
- `ratings.csv`


---

## ⚙️ Installation

Install the required library:

```bash```
pip install scikit-surprise

pip install pandas numpy scikit-learn

🚀 How It Works
Content-Based Filtering

Movie genres are converted into numerical vectors using TF-IDF

Similarity between movies is calculated using cosine similarity

Recommends movies similar to a selected title

Collaborative Filtering (SVD)

Trains on user ratings using Surprise SVD

Predicts ratings for unseen movies

Model performance evaluated using RMSE

Hybrid Recommendation

The final score is calculated as:

Hybrid Score = α × Content Similarity + (1 − α) × Collaborative Score

Where α controls the balance between content-based and collaborative filtering.

▶️ Running the Project
python recommendation_engine.py

🧪 Example Usage
Content-Based Recommendation
content_recommend("Toy Story (1995)")

Collaborative Recommendation
collaborative_recommend(user_id=1)

Hybrid Recommendation
hybrid_recommend(user_id=1, title="Toy Story (1995)")

🎯 Customization

Change number of recommendations:

top_n=10


Adjust hybrid weighting:

alpha=0.7

📈 Evaluation Metric

RMSE (Root Mean Square Error) is used to evaluate the collaborative filtering model

🔮 Future Enhancements

Add movie descriptions, tags, and metadata

Implement deep learning–based recommenders

Handle cold-start users

Build a web interface using Flask or Streamlit

📜 License

This project is intended for educational purposes.

👤 Author

KS Ankith
Aspiring AI & ML Engineer
