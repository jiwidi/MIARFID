import pandas as pd

# Genres
genres = pd.read_csv("data/genre.txt", names=["genre_id", "genre_name"], sep="\t")

# Read movies
all_genre_aux = genres.genre_name.values.tolist()
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
    films_df = films_df[["movie_id", "title", "genres"]]

    films_df.to_csv("data/processed/movies.csv", index=False)


if __name__ == "__main__":
    main()
