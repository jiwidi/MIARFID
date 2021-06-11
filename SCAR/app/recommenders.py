import random
import re
from os import sep

import numpy as np
import pandas as pd
import requests
import tqdm
from scipy.stats.stats import pearsonr

from surprise import SVD, Dataset, KNNBasic, Reader
from surprise.model_selection import cross_validate
from collections import defaultdict

# API FOR POSTERS
api = "591cd379"

# READ DATA
# Genres
generos = pd.read_csv("../data/genre.txt", names=["genre_id", "genre_name"], sep="\t")


# Read users
users_df = pd.read_csv(
    "../data/users.txt", names=["user_id", "age", "gender", "occupation"], sep="\t"
)

# Read movies
all_genre = generos.genre_name.values.tolist()
all_genre = ["movie_id"] + all_genre + ["title"]
films = []
films_df = pd.read_csv("../data/items.txt", names=all_genre, sep="\t")

# Id to title dictionary
movie_id_title = {}
for idx, row in films_df.iterrows():
    movie_id_title[row.movie_id] = row.title

# Ratings
ratings = pd.read_csv(
    "../data/u1_base.txt", names=["user_id", "movie_id", "rating"], sep="\t"
)

ratings = ratings.merge(users_df, on="user_id")
print(f"Data: {len(users_df)} users, {len(films_df)} movies, {len(ratings)} ratings")


# -------------------------
# Demographic recos
def recommend_me_demographic(sex, occupation, lower_age, upper_age, n):
    aux_ratings = ratings[ratings["occupation"] == occupation]
    aux_ratings = aux_ratings[aux_ratings["gender"] == sex]
    aux_ratings = aux_ratings[aux_ratings["age"] < upper_age]
    aux_ratings = aux_ratings[aux_ratings["age"] > lower_age]

    aux_ratings = (
        aux_ratings[["movie_id", "rating"]]
        .groupby("movie_id")
        .agg(count=("rating", "count"), mean=("rating", "mean"))
        .reset_index()
    )
    C = aux_ratings["mean"].mean()
    M = aux_ratings["count"].quantile(0.9)

    def weighted_rating(x):
        v = x["count"]
        R = x["mean"]
        # Calculation based on the IMDB formula
        return (v / (v + M) * R) + (M / (M + v) * C)

    if aux_ratings.empty:
        aux_ratings = ratings.merge(films_df, on="movie_id")
        return random.sample(set(aux_ratings["title"].values), 5)

    aux_ratings["score"] = aux_ratings.apply(weighted_rating, axis=1)
    aux_ratings = aux_ratings.merge(films_df, on="movie_id")
    # print(aux_ratings.sort_values("score", ascending=False))
    return aux_ratings.sort_values("score")["title"].values[:n]


def query_poster(recos):
    poster = []
    fails = []
    for reco in recos:
        # Fetch Movie Data with Full Plot
        title = re.sub(r"\(.*?\)", "", reco).strip()
        if ", The" in title:
            title = "The " + title[:-5]
        elif ", A" in title:
            title = "A  " + title[:-3]
        elif ", Il" in title:
            title = "Il " + title[:-4]

        year = reco[-5:-1]
        params = {"t": title.strip(), "type": "movie", "y": year}
        response = requests.get(
            f"https://www.omdbapi.com/?apikey={api}", params=params
        ).json()
        try:
            poster.append(response["Poster"])
        except:
            print(title, year)
            poster.append("")
            fails.append(title)

    return poster, fails


##################
# Collaborative
# Prepare data


reader = Reader()
data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)
# We precompute recos for all users
predictions = svd.test(trainset.build_testset())


def recommend_me_collaborative(user_id, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [(movie_id_title[x[0]]) for x in user_ratings[:n]]
    return top_n[user_id]


## Hybrid recommendations
# We use both demographic and hybrid


def recommend_me_hybrid(sexo, occ, lower_age, upper_age, user_id, n=10):
    demographic = recommend_me_demographic(sexo, occ, lower_age, upper_age, 50)
    collaborative = recommend_me_collaborative(user_id, 50)

    # Join both recomendations in order, take the first demo/collab recommendations until we achieve 5
    results = []
    c_demo = 0
    c_collab = 0
    for demo, collab in zip(demographic, collaborative):
        if len(results) < 5:
            if demo != collab:
                if c_demo < c_collab:
                    results.append(demographic[c_demo])
                    c_demo += 1
                else:
                    results.append(collaborative[c_collab])
                    c_collab += 1
            else:
                break

    return results
