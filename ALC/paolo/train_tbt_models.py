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

def select_model(model):
    """
    Selects a model
    """
    if model == 0:
        modelo = GradientBoostingClassifier(loss="deviance", learning_rate=0.1, n_estimators=150, verbose=0)
    elif model == 1:
        modelo = svm.LinearSVC(C=100, tol=0.01, loss='hinge', max_iter=500)
    elif model == 2:
        modelo = svm.SVC(C=1)
    elif model == 3:
        modelo = SGDClassifier(alpha=1e-5, loss='squared_hinge', n_jobs=-1, tol=20)
    elif model == 4:
        modelo = MLPClassifier(hidden_layer_sizes=(16),
                                solver='sgd',
                                alpha=0.0001, 
                                batch_size=100, 
                                learning_rate='constant', 
                                learning_rate_init=0.1, 
                                max_iter=500 
                                verbose=True, 
                                warm_start=True,
                                activation='logistic')
    elif model == 5:
        modelo = neighbors.KNeighborsClassifier()
    elif model == 6:
        modelo = RandomForestClassifier(n_estimators=100)
    elif model == 7:
        modelo = GaussianNB()
    elif model == 8:
        modelo = DecisionTreeClassifier()

    return modelo

if __name__ == "__main__":

    # X_train_en, Y_train_en = load_data('dataset/pan21-author-profiling-training-2021-03-14/en')
    # X_train_es, Y_train_es = load_data('dataset/pan21-author-profiling-training-2021-03-14/es')

    parser = argparse.ArgumentParser(usage="python train_models.py --lan [\'es\',\'en\'] --vec [0-1] --model [0-8]")
    parser.add_argument("--lan", type=str, metavar="string -> language, default: en", default='en')
    parser.add_argument("--vec", type=int, metavar="int -> type of vectorizer, default: 0", default=0)
    parser.add_argument("--model", type=int, metavar="int -> type of model, default: 0", default=0)
    args = parser.parse_args()

    if args.lan == 'en':
        #X_train, Y_train = load_data('dataset/pan21-author-profiling-training-2021-03-14/en')
        #X_train, Y_train = process_csv('dataset/data_en.csv', lan = 'en')
        with open('dataset/processed_text_en_tbt.pkl','rb') as f:
            X_train, Y_train = pickle.load(f)

    elif args.lan == 'es':
        #X_train, Y_train = process_csv('dataset/data_es.csv', lan = 'es')
        with open('dataset/processed_text_es_tbt.pkl','rb') as f:
            X_train, Y_train = pickle.load(f)
    
    if args.vec == 0:
        vectorizador = CountVectorizer(tokenizer=casual_tokenize, max_df=0.8)
    elif args.vec == 1:
        vectorizador = TfidfVectorizer(tokenizer=casual_tokenize, max_features=5000, max_df=0.8, ngram_range=(2,3))

    vectorizador.fit(X_train)

    matriz_train = vectorizador.transform(X_train).toarray()


    # Train with custom cross-validation
    ratio = 0.5
    cr_accuracies = []
    cv = 1
    for i in range(0,len(matriz_train)-3999, 4000):
        print("Cross-validation step ", cv, '/10', sep='')
        X_cr_train = numpy.concatenate([matriz_train[:i], matriz_train[i+4000:]])
        Y_cr_train = Y_train[:i] + Y_train[i+4000:]
        X_cr_val = matriz_train[i:i+4000]
        Y_cr_val = Y_train[i:i+4000]
        
        modelo = select_model(args.model)
        modelo.fit(X_cr_train, Y_cr_train)
        pred = modelo.predict(X_cr_val)

        errors = 0
        samples = 0
        for y in range(0, len(Y_cr_val)-199, 200):
            y_pred = pred[y:y+200]
            y_true = Y_cr_val[y+1]
            positive_labels = (y_pred==0).sum()
            negative_labels = (y_pred==1).sum()
            print(positive_labels, negative_labels, y_true)

            # Label full author
            if negative_labels > ratio * positive_labels:
                y_pred = 1
            else:
                y_pred = 0

            # Compare labels
            if y_true != y_pred: 
                errors +=1

            samples += 1
        #
        cr_accuracies.append(1 - errors/samples)
        cv += 1

    print(cr_accuracies)
    print("Model %s: Accuracy (%s): %0.2f (+/- %0.2f)" % ( args.model, args.lan, numpy.array(cr_accuracies).mean(), numpy.array(cr_accuracies).std()))

    quit()
