from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
import xgboost as xgb


def removeStopwords(texto):
    blob = TextBlob(texto).words
    outputlist = [word for word in blob if word not in stopwords.words("spanish")]
    return " ".join(word for word in outputlist)


if __name__ == "__main__":
    print(
        removeStopwords(
            "Todo lo que sigue son ejemplos de acrónimos que no se deberían separar: EE.UU., S.L., CC.OO., S.A., D., U.R.S.S., entre otros."
        )
    )


class XGBoostClassifier:
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({"objective": "multi:softprob"})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(
            params=self.params, dtrain=dtrain, num_boost_round=num_boost_round
        )

    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if "num_boost_round" in params:
            self.num_boost_round = params.pop("num_boost_round")
        if "objective" in params:
            del params["objective"]
        self.params.update(params)
        return self
