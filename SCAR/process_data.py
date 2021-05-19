import pandas as pd

# Genres
genres = pd.read_csv("data/genre.txt", names=["genre_id", "genre_name"], sep="\t")

# Read movies
all_genre_aux = genres.genre_name.values.tolist()
print(all_genre_aux)
users_df = pd.read_csv(
    "data/users.txt", names=["user_id", "edad", "sexo", "occupation"], sep="\t"
)


def condense_genre(row):
    r = []
    for idx, genre in enumerate(all_genre_aux):
        if row[genre] != 0:
            r.append({"id": idx, "name": genre})
    return r


def main():

    all_genre = ["movie_id"] + all_genre_aux + ["title"]
    films = []
    films_df = pd.read_csv("data/items.txt", names=all_genre, sep="\t")
    films_df["genres"] = films_df.apply(condense_genre, axis=1)

    # # ratings
    # ratings = pd.read_csv(
    #     "data/u1_base.txt", names=["user_id", "movie_id", "rating"], sep="\t"
    # )
    # aux_ratings = (
    #     ratings[["movie_id", "rating"]]
    #     .groupby("movie_id")
    #     .agg(rating_count=("rating", "count"), avg_rating=("rating", "mean"))
    #     .reset_index()
    # )[["movie_id", "rating_count", "avg_rating"]]
    # print(aux_ratings)
    # cols_to_use = ratings.columns.difference(films_df.columns)
    # films_df = films_df.merge(aux_ratings, on="movie_id")[
    #     ["movie_id", "rating_count", "avg_rating", "title", "genres"]
    # ]

    films_df = films_df[["movie_id", "title", "genres"]]
    # Demographic scores

    films_df.to_csv("data/processed/movies.csv", index=False)


if __name__ == "__main__":
    main()
