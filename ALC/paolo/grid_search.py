import os
import sys
import argparse
import pickle
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

    parser = argparse.ArgumentParser(usage="python grid_search.py --lan [\'es\',\'en\'] --vec [0-1] --model [0-8]")
    parser.add_argument("--lan", type=str, metavar="string -> language, default: en", default='en')
    parser.add_argument("--vec", type=int, metavar="int -> type of vectorizer, default: 0", default=0)
    parser.add_argument("--model", type=int, metavar="int -> type of model, default: 0", default=0)
    args = parser.parse_args()

    if args.lan == 'en':
        with open('dataset/processed_text_en.pkl','rb') as f:
            X_train, Y_train = pickle.load(f)

    elif args.lan == 'es':
        with open('dataset/processed_text_es.pkl','rb') as f:
            X_train, Y_train = pickle.load(f)
    
    if args.vec == 0:
        vectorizador = CountVectorizer(tokenizer=casual_tokenize, max_df=0.8)
    elif args.vec == 1:
        vectorizador = TfidfVectorizer(tokenizer=casual_tokenize, max_features=5000, max_df=0.8, ngram_range=(2,3))

    vectorizador.fit(X_train)

    matriz_train = vectorizador.transform(X_train)

    if args.model == 0:
        modelo = GradientBoostingClassifier()
        params = {
            'loss':['deviance', 'exponential'],
            'learning_rate':[0.1, 0.01, 0.001, 0.0001],
            'n_estimators':[150, 250, 350, 450, 550]
        }

    elif args.model == 1:
        modelo = svm.LinearSVC(C=100, tol=0.01, loss='hinge', max_iter=500)
        params = {
            'loss':['hinge', 'squared_hinge'],
            'C':[1, 10, 30, 50, 70, 90, 100],
            'tol':[0.01, 0.05, 0.1, 0.2, 0.5, 2, 10, 20],
            'max_iter':[1000]
        }

    elif args.model == 2:
        modelo = svm.SVC(C=1)
        params = {
            'C':[1, 10, 30, 50, 70, 90, 100],
            'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
            'tol':[0.01, 0.05, 0.1, 0.2, 0.5, 2, 10, 20],
            'max_iter':[1000]
        }

    elif args.model == 3:
        modelo = SGDClassifier()
        params = {
            'loss':['hinge', 'squared_hinge', 'perceptron'],
            'alpha':[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
            'tol':[0.01, 0.05, 0.1, 0.2, 0.5, 2, 10, 20],
            'n_jobs':[-1]
        }  

    elif args.model == 4:
        modelo = MLPClassifier(hidden_layer_sizes=(16,32,16),
                                solver='adam',
                                alpha=0.0001, 
                                batch_size=200, 
                                learning_rate='constant', 
                                learning_rate_init=0.001, 
                                random_state=1, 
                                max_iter=500, 
                                verbose=False, 
                                warm_start=True)
        params = {
            'hidden_layer_sizes':[(16,32,16), (16), (16,16), (32), (1024, 64, 16), (1024), (64, 16)],
            'solver':['adam', 'sgd'],
            'activation':['logistic','relu'],
            'learning_rate':['constant','adaptive'],
            'learning_rate_init':[0.1, 0.01, 0.001, 0.0001, 0.00001],
            'max_iter':[500],
            'warm_start':[True,False]
        }

    elif args.model == 5:
        modelo = neighbors.KNeighborsClassifier()
        params = {
            'n_neighbors':[5, 10, 20],
            'n_jobs':[-1]
        }

    elif args.model == 6:
        modelo = RandomForestClassifier(n_estimators=100)
        params = {
            'n_estimators':[100, 150, 300],
            'n_jobs':[-1]
        }

    elif args.model == 7:
        modelo = GaussianNB()
        params = {
            'var_smoothing':[1e-9]
        }

    elif args.model == 8:
        modelo = DecisionTreeClassifier()
        params = {
            'splitter':['best','random']
        }


    grid_search = GridSearchCV(modelo, params, scoring = 'accuracy', cv = 10, n_jobs=-1)
    grid_search.fit(matriz_train.toarray(), Y_train)

    print("Model %s: Accuracy (%s): %0.2f \tBest params: %s" % ( args.model, args.lan, grid_search.best_score_, grid_search.best_params_))

    quit()
