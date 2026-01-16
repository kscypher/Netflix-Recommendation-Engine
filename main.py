# ==============================
# Netflix-Style Recommendation Engine
# Content-Based + Collaborative + Hybrid
# ==============================

# Install the surprise library if it's not already installed
!pip install scikit-surprise

import pandas as pd
import numpy as np

# ------------------------------
# Load Data
# ------------------------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# ------------------------------
# Content-Based Filtering
# ------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies["genres"] = movies["genres"].replace("(no genres listed)", "")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_index = pd.Series(movies.index, index=movies["title"])

def content_recommend(title, top_n=5):
    idx = movie_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["title", "genres"]]

# ------------------------------
# Collaborative Filtering (SVD)
# ------------------------------
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

svd = SVD(n_factors=50, random_state=42)
svd.fit(trainset)

print("Model RMSE:")
rmse(svd.test(testset))

def collaborative_recommend(user_id, top_n=5):
    watched = ratings[ratings["userId"] == user_id]["movieId"].values
    predictions = []

    for movie_id in movies["movieId"]:
        if movie_id not in watched:
            pred = svd.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:top_n]

    return movies[movies["movieId"].isin(
        [m[0] for m in top_movies]
    )][["title", "genres"]]

# ------------------------------
# Hybrid Recommendation
# ------------------------------
def hybrid_recommend(user_id, title, top_n=5, alpha=0.5):
    idx = movie_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    hybrid_scores = []
    for i, sim in sim_scores:
        movie_id = movies.iloc[i]["movieId"]
        try:
            cf_score = svd.predict(user_id, movie_id).est
        except:
            cf_score = 0

        score = alpha * sim + (1 - alpha) * cf_score
        hybrid_scores.append((i, score))

    hybrid_scores = sorted(
        hybrid_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]

    movie_indices = [i[0] for i in hybrid_scores]
    return movies.iloc[movie_indices][["title", "genres"]]

# ------------------------------
# Demo Run
# ------------------------------
if __name__ == "__main__":
    print("\nContent-Based Recommendations:")
    print(content_recommend("Toy Story (1995)"))

    print("\nCollaborative Recommendations (User 1):")
    print(collaborative_recommend(user_id=1))
    

    print("\nHybrid Recommendations (User 1 + Toy Story):")
    print(hybrid_recommend(user_id=1, title="Toy Story (1995)"))
