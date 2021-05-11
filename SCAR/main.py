from os import sep
import pandas as pd
import numpy as np
from classes import User, Movie


# Genres
generos = pd.read_csv("data/genre.txt", names=["genre_id", "genre_name"], sep="\t")


# Read users
users = []
users_df = pd.read_csv(
    "data/users.txt", names=["user_id", "edad", "sexo", "occupation"], sep="\t"
)
for idx, row in users_df.iterrows():
    users.append(User(row["user_id"], row["edad"], row["sexo"], row["occupation"]))

# Read movies
all_genre = generos.genre_name.values.tolist()
all_genre = ["movie_id"] + all_genre + ["title"]
films = []
films_df = pd.read_csv("data/items.txt", names=all_genre, sep="\t")
for idx, row in films_df.iterrows():
    films.append(Movie(row["movie_id"], row[all_genre].values.tolist(), row["title"]))


# Ratings
ratings = pd.read_csv(
    "data/u1_base.txt", names=["user_id", "movie_id", "rating"], sep="\t"
)

ratings = ratings.merge(users_df, on="user_id")
print(len(ratings))

# Demographic recos
def recommend_me_demographic(user):
    occupation = user.occupation

    # Search for films seen by this profession
    aux_ratings = ratings[ratings["occupation"] == occupation]
    aux_ratings = (
        aux_ratings[["movie_id", "rating"]]
        .groupby("movie_id")
        .agg(count=("rating", "count"), mean=("rating", "mean"))
        .reset_index()
    )
    C = aux_ratings["mean"].mean()
    M = aux_ratings["count"].quantile(0.9)

    def weighted_rating(x, m=M, C=C):
        v = x["count"]
        R = x["mean"]
        # Calculation based on the IMDB formula
        return (v / (v + m) * R) + (m / (m + v) * C)

    aux_ratings["score"] = aux_ratings.apply(weighted_rating, axis=1)

    # print(aux_ratings.sort_values("score", ascending=False))
    return aux_ratings.sort_values("score")


recommend_me_demographic(users[5])


# Collaborative recos
# Preferences matrix
preference_matrix = (
    ratings[["user_id", "movie_id"]]
    .merge(films_df, on="movie_id")
    .groupby("user_id")
    .mean()
)
print(preference_matrix)

items_matrix = (
    ratings.pivot(index="user_id", columns="movie_id", values="rating")
    .reset_index()
    .fillna(0)
)


print(items_matrix)
