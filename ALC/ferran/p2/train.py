import numpy as np
import pandas as pd
from nltk.tokenize.casual import casual_tokenize

# Models
from sklearn import neighbors, svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

RANDOM_STATE = 17


def train_evaluate(model, params, train_matrix, dev_matrix, train_target, dev_target):
    grid_search = GridSearchCV(
        estimator=model, param_grid=params[0], scoring="f1_macro", n_jobs=-1, cv=5
    )
    grid_search.fit(train_matrix, train_target)

    print(
        f"Model {type(model).__name__}: F1_macro {grid_search.best_score_:.2f} \tBest params: {grid_search.best_params_}"
    )
    return type(model).__name__, grid_search.best_score_

    # print("Finished.")

    # print("accuracy = ", accuracy_score(dev_target, out))
    # print(
    #     "macro = ", precision_recall_fscore_support(dev_target, out, average="macro"),
    # )
    # print(
    #     "micro = ", precision_recall_fscore_support(dev_target, out, average="micro"),
    # )
    # print(classification_report(dev_target, out))


if __name__ == "__main__":
    train = pd.read_csv("data/TASS2017_T1_training_parsed.csv")
    dev = pd.read_csv("data/TASS2017_T1_development_parsed.csv")
    test = pd.read_csv("data/TASS2017_T1_test_parsed.csv")
    ##Apply regex to clean columns
    # train["text"] = train["text"].str.replace(r"@+\w+' + '|' + '#+\w+", "")
    # train["text"] = train["text"].str.replace(r"\D+", "")
    ##Vectorizing columns
    print("Vectorizing...")
    vect = TfidfVectorizer(tokenizer=casual_tokenize, max_df=0.8)
    vect.fit(train["text"])
    vect.fit(dev["text"])

    train_matrix = vect.transform(train["text"]).toarray()
    dev_matrix = vect.transform(dev["text"]).toarray()
    print("Vectorized.")

    models = [
        svm.SVC(),
        svm.LinearSVC(),
        GaussianNB(),
        GradientBoostingClassifier(),
        SGDClassifier(),
        neighbors.KNeighborsClassifier(),
    ]

    parameters = [
        [  # SVC
            {
                "kernel": ["rbf", "linear"],
                "gamma": [1e-3, 1e-4],
                "C": [1, 10, 100, 1000],
                "random_state": [RANDOM_STATE],
            },
        ],
        [
            {
                "C": [1, 10, 100, 1000],
                "random_state": [RANDOM_STATE],
                "max_iter": [10000000],
            }
        ],  # LinearSVC
        [{}],  # GaussianNB
        [  # GradientBoostingClassifier
            {
                "loss": ["deviance"],
                "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                "min_samples_split": np.linspace(0.1, 0.5, 12),
                "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                "max_depth": [3, 5, 8],
                "max_features": ["log2", "sqrt"],
                "criterion": ["friedman_mse", "mae"],
                "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                "n_estimators": list(range(1, 10000, 100)),
            }
        ],
        [  # SGDClassifier
            {
                "loss": ["hinge", "log", "squared_hinge", "modified_huber"],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "penalty": ["l2", "l1", "none"],
                "max_iter": [10000000],
            }
        ],
        [  # Kneighbors]
            {
                "n_neighbors": list(range(1, 10, 1)),
                "leaf_size": list(range(20, 40, 1)),
                "p": [1, 2],
                "weights": ["uniform", "distance"],
                "metric": ["minkowski", "chebyshev"],
            }
        ],
    ]

    names = []
    scores = []
    for model, param in zip(models, parameters):
        print(f"Evaluating model {type(model).__name__}")
        name, score = train_evaluate(
            model, param, train_matrix, dev_matrix, train["target"], dev["target"]
        )
        names.append(name)
        scores.append(score)
    df = pd.DataFrame({"model_name": names, "f1_macro": scores})
    df.to_csv("results.csv", index=False)

