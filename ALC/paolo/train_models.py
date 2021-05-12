import os
import sys
import argparse
import pickle
from typing import TextIO
from xml.dom import minidom

import numpy
from nltk.tokenize.casual import casual_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
#from zeugma.embeddings import EmbeddingTransformer

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

    parser = argparse.ArgumentParser(usage="python train_models.py --lan [\'es\',\'en\'] --vec [0-1] --model [0-8]")
    parser.add_argument("--lan", type=str, metavar="string -> language, default: en", default='en')
    parser.add_argument("--vec", type=int, metavar="int -> type of vectorizer, default: 0", default=0)
    parser.add_argument("--model", type=int, metavar="int -> type of model, default: 0", default=0)
    parser.add_argument("--test", type=bool, metavar="bool -> do inference with test set, default: False", default=False)
    args = parser.parse_args()

    if args.lan == 'en':
        #X_train, Y_train = load_data('dataset/pan21-author-profiling-training-2021-03-14/en')
        #X_train, Y_train = process_csv('dataset/data_en.csv', lan = 'en')
        with open('dataset/processed_text_en.pkl','rb') as f:
            X_train, Y_train = pickle.load(f)

    elif args.lan == 'es':
        #X_train, Y_train = process_csv('dataset/data_es.csv', lan = 'es')
        with open('dataset/processed_text_es.pkl','rb') as f:
            X_train, Y_train = pickle.load(f)
    
    if args.vec == 0:
        vectorizador = CountVectorizer(tokenizer=casual_tokenize, max_df=0.8)
    elif args.vec == 1:
        vectorizador = TfidfVectorizer(tokenizer=casual_tokenize, max_features=5000, max_df=0.8, ngram_range=(2,3))

    vectorizador.fit(X_train)

    matriz_train = vectorizador.transform(X_train)

    if args.model == 0:
        modelo = GradientBoostingClassifier(loss="deviance", learning_rate=0.1, n_estimators=150, verbose=0)
    elif args.model == 1:
        modelo = svm.LinearSVC(C=100, tol=0.1, loss='hinge', max_iter=1000000)
    elif args.model == 2:
        modelo = svm.SVC(C=1)
    elif args.model == 3:
        modelo = SGDClassifier(alpha=1e-5, loss='squared_hinge', n_jobs=-1, tol=20 )
    elif args.model == 4:
        modelo = MLPClassifier(hidden_layer_sizes=(16),
                                solver='adam',
                                alpha=0.0001, 
                                batch_size=100, 
                                learning_rate='constant', 
                                learning_rate_init=1e-5, 
                                max_iter=1000, 
                                verbose=True, 
                                warm_start=True,
                                activation='relu')
    elif args.model == 5:
        modelo = neighbors.KNeighborsClassifier()
    elif args.model == 6:
        modelo = RandomForestClassifier(n_estimators=100)
    elif args.model == 7:
        modelo = GaussianNB()
    elif args.model == 8:
        modelo = DecisionTreeClassifier()

    if not args.test:
        scores = cross_val_score(
            modelo, matriz_train.toarray(), Y_train, cv=10, scoring="accuracy"
        )
        print("Model %s: Accuracy (%s): %0.2f (+/- %0.2f)" % ( args.model, args.lan, scores.mean(), scores.std() * 2))

    if args.test:
        modelo.fit(matriz_train.toarray(), Y_train)
        print("Training accuracy: ", modelo.score(matriz_train.toarray(), Y_train))
        print("Predicting the test samples")
        if args.lan == 'en':
            with open('dataset/processed_text_en_test.pkl','rb') as f:
                X_test, User_ids = pickle.load(f)

        elif args.lan == 'es':
            with open('dataset/processed_text_es_test.pkl','rb') as f:
                X_test, User_ids = pickle.load(f)

        matriz_test = vectorizador.transform(X_test)
        Y_pred = modelo.predict(matriz_test.toarray())
        assert len(Y_pred) == len(User_ids)
        for i in range(len(User_ids)):
            with open('predictions/' + args.lan + '/' + User_ids[i] + '.xml', 'w') as f:
                f.write('<author id=\"' + User_ids[i] + '\"\n    lang=\"' + args.lan + '\"\n    type=\"' + str(Y_pred[i]) + '\"\n/>')
                f.close()
    quit()
