import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer

# Models
from sklearn import neighbors, svm
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from utils import XGBoostClassifier

RANDOM_STATE = 17


def train_evaluate(model, params, train_matrix, train_target):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params[0],
        scoring="f1_macro",
        n_jobs=-1,
        cv=5,
        verbose=10,
    )
    grid_search.fit(train_matrix, train_target)

    print(
        f"Model {type(model).__name__}: F1_macro {grid_search.best_score_:.2f} \tBest params: {grid_search.best_params_}"
    )
    return type(model).__name__, grid_search.best_score_, grid_search.best_estimator_


if __name__ == "__main__":
    train = pd.read_csv("data/TASS2017_T1_training_parsed.csv")
    dev = pd.read_csv("data/TASS2017_T1_development_parsed.csv")

    train = pd.concat([train, dev])

    test = pd.read_csv("data/TASS2017_T1_test_parsed.csv")

    # Tokenizer to clean columns
    tokenizer = TweetTokenizer(
        strip_handles=False, reduce_len=True, preserve_case=False
    )

    train["text"] = list(map(" ".join, map(tokenizer.tokenize, train["text"])))
    test["text"] = list(map(" ".join, map(tokenizer.tokenize, test["text"])))

    # Vectorizing columns
    print("Vectorizing...")
    vect = TfidfVectorizer(max_df=0.3, analyzer="char_wb", min_df=2, ngram_range=(4, 5))

    vect.fit(train["text"])

    train_matrix = vect.transform(train["text"]).toarray()
    print("Vectorized.")

    models = [
        RandomForestClassifier(),
        XGBoostClassifier(
            eval_metric="auc",
            num_class=len(train["target"].unique()),
            nthread=8,
        ),
        svm.SVC(),
        svm.LinearSVC(),
        GaussianNB(),
        GradientBoostingClassifier(),
        SGDClassifier(),
        neighbors.KNeighborsClassifier(),
        MLPClassifier(),
    ]

    parameters = [
        [  # RandomForest
            {
                "bootstrap": [True, False],
                "max_features": ["auto", "sqrt"],
                "n_estimators": [
                    200,
                    800,
                    1600,
                    2000,
                ],
            }
        ],
        [  # XGBOOST
            {
                "num_boost_round": [500],
                "eta": [0.05, 0.1],
                "max_depth": [12],
            }
        ],
        [  # SVC
            {
                "kernel": ["rbf", "linear"],
                "gamma": [1e-3, 1e-4],
                "C": [10, 1000, 10000],
                "random_state": [RANDOM_STATE],
            },
        ],
        [  # LinearSVC
            {
                "C": [1, 10, 100, 1000],
                "random_state": [RANDOM_STATE],
                "max_iter": [10000000],
            }
        ],
        [{}],  # GaussianNB
        [  # GradientBoostingClassifier
            {
                "loss": ["deviance"],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 8],
                "max_features": ["log2", "sqrt"],
                "criterion": ["friedman_mse"],  # ["friedman_mse", "mae"],
                "subsample": [0.3, 0.5, 0.7],
                "n_estimators": [400],  # list(range(1, 401, 200)),
                "random_state": [RANDOM_STATE],
            }
        ],
        [  # SGDClassifier
            {
                "loss": ["hinge", "log", "modified_huber"],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "penalty": ["l1", "none"],
                "max_iter": [10000000],
                "random_state": [RANDOM_STATE],
            }
        ],
        [  # Kneighbors]
            {
                "n_neighbors": list(range(3, 10, 1)),
                "leaf_size": list(range(20, 40, 1)),
                "p": [2],
                "weights": ["distance"],
                "metric": ["minkowski"],
            }
        ],
        [  # MLP
            {
                "activation": ["relu"],
                "alpha": [1e-05],
                "hidden_layer_sizes": [
                    (16, 16),
                    (32),
                    (32, 32),
                    (64, 64),
                ],
                "learning_rate": ["constant"],
                "learning_rate_init": [0.001, 0.01, 0.1],
                "solver": ["sgd", "adam"],
                "random_state": [RANDOM_STATE],
                "max_iter": [1000],
                "batch_size": [32],
            }
        ],
    ]
    print(f"{len(models)}, {len(parameters)}")
    names = []
    scores = []
    estimators = {}
    best_score = 0
    best_model = None
    best_name = None
    best_params = None

    for model, param in zip(models, parameters):
        print(f"Evaluating model {type(model).__name__}")
        name, score, best_model_tmp = train_evaluate(
            model, param, train_matrix, train["target"]
        )
        names.append(name)
        scores.append(score)
        if score > best_score:
            best_model = best_model_tmp
            best_name = name
            best_params = param
            best_score = score

        if score > 0.40:
            estimators[name] = best_model_tmp

    # Default models
    base_scores = []
    for model in models:
        print(f"Evaluating model {type(model).__name__} without params")
        name, score, best_model_tmp = train_evaluate(
            model, [{}], train_matrix, train["target"]
        )
        base_scores.append(score)

    df = pd.DataFrame(
        {"model_name": names, "default": base_scores, "optimized": scores}
    ).round(2)
    df.to_csv("results.csv", index=False)
    df.to_latex("results.tex", index=False)

    # Predict test set with best_model.
    print(f"Saving predicitions of best model {best_name}")
    test_matrix = vect.transform(test["text"]).toarray()
    test["target"] = best_model.predict(test_matrix)
    test[["id", "target"]].to_csv("data/predict_test.csv", sep="\t", index=False)

    print(f"Ensembling top estimators")
    ensemble_estimators = [(key, value) for key, value in estimators.items()]
    eclf1 = VotingClassifier(estimators=ensemble_estimators, voting="hard")
    scores = cross_val_score(
        eclf1, train_matrix, train["target"], cv=3, scoring="f1_macro", verbose=5
    )
    print(f"Ensemble score {np.mean(scores)}")
    if (np.mean(scores)) > best_score:
        print(f"Saving predicitions of best model {eclf1}")
        eclf1 = VotingClassifier(estimators=ensemble_estimators, voting="hard")
        eclf1.fit(train_matrix, train["target"])
        test["target"] = eclf1.predict(test_matrix)
        test[["id", "target"]].to_csv("data/predict_test.csv", sep="\t", index=False)
