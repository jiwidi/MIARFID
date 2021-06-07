from flask import Flask, render_template, request
from flask import render_template, url_for
import flask
import json
from recommenders import *

app = Flask(__name__)

default_reco = {
    "rec_one": "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.booooooom.com%2Fwp-content%2Fuploads%2F2016%2F09%2Ftextless-movieposters37.jpg&f=1&nofb=1"
}

sample_img = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.booooooom.com%2Fwp-content%2Fuploads%2F2016%2F09%2Ftextless-movieposters37.jpg&f=1&nofb=1"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demografico")
def demografico():
    return render_template("demografico.html", data=default_reco)


@app.route("/colaborativo")
def colaborativo():
    return render_template("colaborativo.html", data=default_reco)


@app.route("/hibrido")
def hibrido():
    return render_template("hibrido.html", data=default_reco)


@app.route("/rec_demo", methods=["GET", "POST"])
def rec_demo():
    sexo = request.form.get("sexo", "F")
    occ = request.form.get("work", "writer")
    age = request.form.get("age", "21-30")
    lower_age = int(age.split("-")[0])
    upper_age = int(age.split("-")[1])
    reco_titles = recommend_me_demographic(sexo, occ, lower_age, upper_age, 5)
    posters = query_poster(reco_titles)[0]

    print(posters)
    response_data = {
        "rec_one": posters[0],
        "rec_two": posters[1],
        "rec_three": posters[2],
        "rec_four": posters[3],
        "rec_five": posters[4],
        "title_one": reco_titles[0],
        "title_two": reco_titles[1],
        "title_three": reco_titles[2],
        "title_four": reco_titles[3],
        "title_five": reco_titles[4],
    }

    return render_template("demografico.html", data=response_data)


@app.route("/rec_collaborative", methods=["GET", "POST"])
def rec_collaborative():
    print(request.form)
    user_id = request.form.get("user_id", "2")
    user_id = int(user_id)
    reco_titles = recommend_me_collaborative(user_id, 5)
    posters = query_poster(reco_titles)[0]

    print(posters)
    response_data = {
        "rec_one": posters[0],
        "rec_two": posters[1],
        "rec_three": posters[2],
        "rec_four": posters[3],
        "rec_five": posters[4],
        "title_one": reco_titles[0],
        "title_two": reco_titles[1],
        "title_three": reco_titles[2],
        "title_four": reco_titles[3],
        "title_five": reco_titles[4],
    }

    return render_template("colaborativo.html", data=response_data)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
