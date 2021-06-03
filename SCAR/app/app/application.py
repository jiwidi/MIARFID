from flask import Flask, render_template, request
from flask import render_template, url_for
import flask
import json

app = Flask(__name__)

default_reco = {
    "rec_one": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.booooooom.com%2Fwp-content%2Fuploads%2F2016%2F09%2Ftextless-movieposters37.jpg&f=1&nofb=1"
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demografico")
def demografico():

    return render_template("demografico.html", data=default_reco)


@app.route("/rec_demo")
def rec_demo():
    print("de lokos")
    response_data = {"rec_one": "pic_trulli.jpg"}

    return render_template("demografico.html", data=response_data)


@app.route("/recommendation")
def recommend():
    user_input = dict(request.args)
    method_ = list(user_input.values())[-1]
    recommender = Recommender(user_input)
    if method_ == "NMF":
        recommendations = recommender.nmf()
    else:
        recommendations = recommender.cosine()
    imdb_ids_dict = postgres_extract(recommendations.keys())
    for movie_id, imdb_id in imdb_ids_dict.items():
        recommendations[movie_id]["omdb_dict"] = omdb_extract(imdb_id)
    return render_template(
        "recommendation.html", movies=recommendations, input=user_input
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
