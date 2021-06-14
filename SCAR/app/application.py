from flask import Flask, render_template, request
from flask import render_template
from recommenders import (
    recommend_me_collaborative,
    recommend_me_demographic,
    recommend_me_hybrid,
    query_poster,
    get_demo_data_from_user
)
import random
import numpy as np

# SEED
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)

app = Flask(__name__)

sample_img = "https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fwww.technistone.com%2Fcolor-range%2Fimage-product%2FProd%2520Crystal%2520Absolute%2520White.jpg&f=1&nofb=1"


default_reco = {
    "rec_one": sample_img,
    "rec_two": sample_img,
    "rec_three": sample_img,
    "rec_four": sample_img,
    "rec_five": sample_img,
}

latest_user = 999

new_users = {}

@app.route("/")
def index():
    return render_template("registro.html", data=default_reco)

@app.route("/registro_home")
def registro_home():
    return render_template("registro.html", data=default_reco)

@app.route("/registro", methods=["GET", "POST"])
def registro():
    global latest_user
    global new_users

    latest_user+=1
    new_user_id =latest_user

    sexo = request.form.get("sexo", "F")
    occ = request.form.get("work", "writer")
    age = request.form.get("age", "21-30")
    lower_age = int(age.split("-")[0])
    upper_age = int(age.split("-")[1])

    new_users[new_user_id] = (sexo, occ, lower_age, upper_age)
    print(new_users)
    data = {
        "new_user_id":str(new_user_id)
    }
    return render_template("registro.html", data=data)

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
    global new_users

    print(request.form)
    user_id = request.form.get("user_id", "2")
    if(len(user_id)<1):
        user_id=2
    user_id = int(user_id)
    if(user_id<1000):
        sexo, occ, lower_age, upper_age = get_demo_data_from_user(user_id)
    elif user_id in new_users:
        sexo, occ, lower_age, upper_age = new_users[user_id]
        print("EEEEEEEEEEASDASD")
    else:
        return render_template("demografico.html", data=default_reco)

    print("THIS IS DATA", sexo, occ, lower_age, upper_age)
    print(recommend_me_demographic(sexo, occ, lower_age, upper_age, 5))
    reco_titles, reco_ratings = recommend_me_demographic(sexo, occ, lower_age, upper_age, 5)
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
        "rating_one": reco_ratings[0],
        "rating_two": reco_ratings[1],
        "rating_three": reco_ratings[2],
        "rating_four": reco_ratings[3],
        "rating_five": reco_ratings[4],
    }

    return render_template("demografico.html", data=response_data)


@app.route("/rec_collaborative", methods=["GET", "POST"])
def rec_collaborative():
    global new_users

    print(request.form)
    user_id = request.form.get("user_id", "2")
    user_id = int(user_id)
    print("USER,", user_id)
    if(user_id in new_users or user_id>944):
        return render_template("colaborativo.html", data=default_reco)


    reco_titles, reco_ratings = recommend_me_collaborative(user_id, 5)
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
        "rating_one": reco_ratings[0],
        "rating_two": reco_ratings[1],
        "rating_three": reco_ratings[2],
        "rating_four": reco_ratings[3],
        "rating_five": reco_ratings[4],
    }

    return render_template("colaborativo.html", data=response_data)


@app.route("/rec_hybrid", methods=["GET", "POST"])
def rec_hybrid():
    global new_users
    print(request.form)
    user_id = request.form.get("user_id", "2")
    if(len(user_id)<1):
        user_id=2
    user_id = int(user_id)

    if(user_id<944):
        sexo, occ, lower_age, upper_age = get_demo_data_from_user(user_id)
        reco_titles, reco_ratings = recommend_me_hybrid(sexo, occ, lower_age, upper_age, user_id, 5)
    elif(user_id in new_users):
        sexo, occ, lower_age, upper_age = new_users[user_id]
        reco_titles, reco_ratings = recommend_me_demographic(sexo, occ, lower_age, upper_age, 5)
    else:
        return render_template("hibrido.html", data=default_reco)

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
        "rating_one": reco_ratings[0],
        "rating_two": reco_ratings[1],
        "rating_three": reco_ratings[2],
        "rating_four": reco_ratings[3],
        "rating_five": reco_ratings[4],
    }

    return render_template("hibrido.html", data=response_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=True, port=6006)
