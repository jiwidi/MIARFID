import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

# Metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Models
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB


def train_evaluate(model, train_matrix, dev_matrix, train_target, dev_target):
    model.fit(train_matrix, train_target)
    out = model.predict(dev_matrix)
    print("Finished.")

    print("accuracy = ", accuracy_score(dev_target, out))
    print(
        "macro = ", precision_recall_fscore_support(dev_target, out, average="macro"),
    )
    print(
        "micro = ", precision_recall_fscore_support(dev_target, out, average="micro"),
    )
    print(classification_report(dev_target, out))


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
        svm.SVC(C=1, kernel="linear"),
        svm.LinearSVC(C=100, tol=0.01, loss="hinge", max_iter=1000000000),
        GaussianNB(),
        GradientBoostingClassifier(
            n_estimators=1000, learning_rate=0.1, max_depth=3, random_state=17
        ),
        SGDClassifier(),
        neighbors.KNeighborsClassifier(),
    ]

    for model in models:
        print(f"Evaluating model {model}")
        train_evaluate(model, train_matrix, dev_matrix, train["target"], dev["target"])

