import re

import pickle
import demoji
import emoji
import nltk
import pandas as pd
from demoji import replace_with_desc
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

"""

Script for processing Hate Speech tweets for a PAN task

Authors:
    Jaime Ferrando Huertas
    Javier Martínez Bernia

"""


def stem_tweet(tweet, lan):
    """
    Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens.
    """

    if lan == "en":
        stemmer = PorterStemmer()
        # tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
        tokens = [stemmer.stem(t) for t in tweet.split()]
    else:
        stemmer = SnowballStemmer("spanish")
        tokens = [stemmer.stem(t) for t in tweet.split()]

    return " ".join(tokens)


def basic_tokenize(tweet):
    """
    Same as tokenize but without the stemming
    """

    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


def rmv_tags(tweet):
    """
    Removes some tags of the tweet -> '#USER#' '#URL#'
    """

    tweet = tweet.replace("#USER#", "")
    tweet = tweet.replace("#HASHTAG#", "")
    tweet = tweet.replace("HASHTAG#", "")
    tweet = tweet.replace("#URL#", "")
    tweet = tweet.replace("RETWEET", "")
    if tweet.startswith("RT :"):
        tweet = tweet.replace("RT :", "")

    achronims = re.compile(r"([A-Z]{1,2}\.)+")
    symbols = re.compile(r"(\W*)(\w+)(\W*)")
    tweet_2 = []
    for word in tweet.split():
        if achronims.match(word):
            tweet_2.append(achronims.match(word)[0])
        else:
            re_sym = symbols.findall(word)
            if len(re_sym) > 0:
                for item in re_sym:
                    tweet_2.append(item[1])
    tweet = " ".join(tweet_2)

    return tweet


def rmv_stopwords(tweet, lan):
    """
    Remove stopwords
    """

    tweet_tokens = word_tokenize(tweet)
    if lan == "en":
        tweet_tokens = [
            word.lower()
            for word in tweet_tokens
            if (not word in stopwords.words()) and (len(word) > 2)
        ]
    else:
        tweet_tokens = [
            word.lower()
            for word in tweet_tokens
            if (not word in stopwords.words("spanish")) and (len(word) > 2)
        ]

    tweet = " ".join(tweet_tokens)

    return tweet


def demoji_tweet(tweet, lan):
    """
    Substitute emojis with their description
    """
    if lan == "es":
        tweet2 = []
        for word in tweet.split():
            word = emoji.demojize(word, language="es").split("_")
            for w in word:
                tweet2.append(w)
        tweet = " ".join(tweet2)

    else:
        tweet = replace_with_desc(tweet, sep=" ")
    return tweet


def process_csv(filename,
                lan="en",
                stem=True, 
                remove_tags=True, 
                remove_stopwords=True, 
                demoji=True):
    """

    Process a .csv file with the tweets

    :param string filename
        File to process.

    :param string lan
        Language of the text (es/en).

    :param bool stemm
        Flag to indicate whether to apply stemming to the text.

    :param bool remove_tags
        Flag to indicate whether to remove some tags in the text.

    :param bool remove_stopwords
        Flag to indicate whether to remove stopwords.

    :param bool demoji
        Flag to indicate whether to replace emojis with their textual description.

    :return list X, Y
        Two lists with the tweets in X and the labels in Y

    """

    X = []
    Y = []
    data = pd.read_csv(filename)

    for usr in tqdm(data["author_id"].unique()):
        user_tweets = ""
        sub_data = data[data["author_id"] == usr]
        user_label = sub_data["tag"].unique()[0]

        for tweet in sub_data["tweet"]:

            if remove_tags:
                tweet = rmv_tags(tweet)
            #
            if demoji:
                tweet = demoji_tweet(tweet, lan)
            if remove_stopwords:
                tweet = rmv_stopwords(tweet, lan)
            if stem:
                tweet = stem_tweet(tweet, lan)
            user_tweets += tweet + " "
        #
        X.append(user_tweets)
        Y.append(user_label)
    #
    return X, Y

def process_csv_tweet_by_tweet(filename,
                lan="en",
                stem=True, 
                remove_tags=True, 
                remove_stopwords=True, 
                demoji=True):
    """

    Process a .csv file with the tweets

    :param string filename
        File to process.

    :param string lan
        Language of the text (es/en).

    :param bool stemm
        Flag to indicate whether to apply stemming to the text.

    :param bool remove_tags
        Flag to indicate whether to remove some tags in the text.

    :param bool remove_stopwords
        Flag to indicate whether to remove stopwords.

    :param bool demoji
        Flag to indicate whether to replace emojis with their textual description.

    :return list X, Y
        Two lists with the tweets in X and the labels in Y

    """

    X = []
    Y = []
    data = pd.read_csv(filename)

    for usr in tqdm(data["author_id"].unique()):
        user_tweets = []
        sub_data = data[data["author_id"] == usr]
        user_label = sub_data["tag"].unique()[0]

        for tweet in sub_data["tweet"]:

            if remove_tags:
                tweet = rmv_tags(tweet)
            if demoji:
                tweet = demoji_tweet(tweet, lan)
            if remove_stopwords:
                tweet = rmv_stopwords(tweet, lan)
            if stem:
                tweet = stem_tweet(tweet, lan)
            user_tweets.append(tweet)
        #
        for t in user_tweets:
            X.append(t)
            Y.append(user_label)
    #
    return X, Y


if __name__ == "__main__":
    # demoji.download_codes()
    #X, Y = process_csv('dataset/data_es.csv', lan='es')
    #X, Y = process_csv("dataset/data_en.csv", lan="en")
    #with open('processed_text_es_prova.pkl', 'wb') as f:
    #    pickle.dump((X,Y), f)

    X, Y = process_csv_tweet_by_tweet("dataset/data_en.csv", lan="en")
    with open('processed_text_en_tbt.pkl', 'wb') as f:
        pickle.dump((X,Y), f)

    #print(X)
