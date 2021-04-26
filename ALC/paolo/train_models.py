import os
from xml.dom import minidom

import numpy
from nltk.tokenize.casual import casual_tokenize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from zeugma.embeddings import EmbeddingTransformer

from processing import process_csv


def load_data(dataset_path):
    """
    Loads the data from dataset_path.

    :param string dataset_path
        Path of the data

    :return X, Y
        Two lists with the data in X and the labels in Y
    """

    X = []
    Y = []

    with open(dataset_path + "/truth.txt") as truth_file:
        for line in truth_file:
            usr_id, truth = line.split(":::")
            Y.append(truth)

            xml_file = minidom.parse(dataset_path + "/" + usr_id + ".xml")
            tweets = xml_file.getElementsByTagName("document")

            samples = []
            for tweet in tweets:
                samples.append(tweet.firstChild.data)
            samples = " ".join(samples)
            samples = samples.replace("\n", "")
            X.append(samples)

    return X, Y


if __name__ == "__main__":

    # X_train_en, Y_train_en = load_data('dataset/pan21-author-profiling-training-2021-03-14/en')
    # X_train_es, Y_train_es = load_data('dataset/pan21-author-profiling-training-2021-03-14/es')

    # X_train_en, Y_train_en = process_csv('dataset/data_en.csv', lan = 'en')
    X_train_es, Y_train_es = process_csv("dataset/data_es.csv", lan="es")

    # vectorizador = TfidfVectorizer(tokenizer=casual_tokenize, max_features=1000, max_df=0.8, ngram_range=(2,3))
    vectorizador = CountVectorizer(tokenizer=casual_tokenize, max_df=0.8)

    # vectorizador.fit(X_train_en)
    vectorizador.fit(X_train_es)

    # matriz_train_en = vectorizador.transform(X_train_en)
    matriz_train_es = vectorizador.transform(X_train_es)

    # print(len(matriz_train_en.toarray()[1])) // Matriz de 200 muestras de 26080 caracteristicas

    modelo_en = GradientBoostingClassifier(
        loss="deviance", learning_rate=0.01, n_estimators=250, verbose=1
    )
    scores_en = cross_val_score(
        modelo_en, matriz_train_es.toarray(), Y_train_es, cv=10, scoring="accuracy"
    )
    print("Accuracy (es): %0.2f (+/- %0.2f)" % (scores_en.mean(), scores_en.std() * 2))

    quit()
